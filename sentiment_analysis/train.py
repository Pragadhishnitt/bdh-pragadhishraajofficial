"""
Main training script for BDH sentiment analysis.
Supports continual learning evaluation with temporal ordering.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from configs.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from data import SentimentDataLoader
from models import BDHSentiment, BDHSentimentConfig
from evaluation import MetricsCalculator
from utils import set_seed, get_device_and_dtype, save_checkpoint


class Trainer:
    """Trainer class for BDH sentiment analysis."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        set_seed(config.training.seed)
        
        # Setup device and dtype
        self.device, self.dtype, self.ctx, self.scaler = get_device_and_dtype(
            config.training.device,
            config.training.dtype
        )
        
        # Load data
        print("\n" + "="*50)
        print("Loading Data")
        print("="*50)
        self.data_loader = SentimentDataLoader(
            data_path=config.data.data_path,
            tokenizer_name=config.data.tokenizer_name,
            max_seq_length=config.data.max_seq_length,
            num_samples=config.data.num_samples,
            train_ratio=config.data.train_ratio,
            batch_size=config.data.batch_size
        )
        self.train_loader, self.val_loader = self.data_loader.get_dataloaders()
        
        # Initialize model
        print("\n" + "="*50)
        print("Initializing Model")
        print("="*50)
        model_config = BDHSentimentConfig(
            n_layer=config.model.n_layer,
            n_embd=config.model.n_embd,
            n_head=config.model.n_head,
            mlp_internal_dim_multiplier=config.model.mlp_internal_dim_multiplier,
            vocab_size=config.model.vocab_size,
            dropout=config.model.dropout,
            num_classes=config.model.num_classes,
            max_seq_length=config.model.max_seq_length
        )
        self.model = BDHSentiment(model_config).to(self.device)
        
        # Count parameters
        total_params = self.model.count_parameters()
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        if total_params > 17e6:
            print(f"WARNING: Model has {total_params/1e6:.2f}M parameters (> 17M limit)")
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        total_steps = config.training.max_iters
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Metrics
        self.train_metrics = MetricsCalculator(num_classes=config.model.num_classes)
        self.val_metrics = MetricsCalculator(num_classes=config.model.num_classes)
        
        
        # Results storage
        # Convert config dataclass to dict for JSON serialization
        from dataclasses import asdict
        config_dict = {
            'name': config.name,
            'model': asdict(config.model),
            'data': asdict(config.data),
            'training': asdict(config.training),
            'enable_continual_metrics': config.enable_continual_metrics,
            'time_window_size': config.time_window_size,
            'compare_with_baseline': config.compare_with_baseline,
            'baseline_model': config.baseline_model
        }
        self.results = {
            'config': config_dict,
            'train_history': [],
            'val_history': [],
            'best_val_acc': 0.0
        }
        
        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            with self.ctx:
                logits, loss = self.model(input_ids, attention_mask, labels)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            # Update metrics
            self.train_metrics.update(logits, labels, batch.get('timestamp'))
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.train_metrics.compute_standard_metrics()
        metrics['loss'] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, epoch: int):
        """Evaluate on validation set."""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with self.ctx:
                logits, loss = self.model(input_ids, attention_mask, labels)
            
            self.val_metrics.update(logits, labels, batch.get('timestamp'))
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.val_metrics.compute_all_metrics(
            window_size=self.config.time_window_size
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        num_epochs = self.config.training.max_iters // len(self.train_loader) + 1
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.results['train_history'].append(train_metrics)
            
            # Evaluate
            if (epoch + 1) % (self.config.training.eval_interval // len(self.train_loader) + 1) == 0:
                val_metrics = self.evaluate(epoch)
                self.results['val_history'].append(val_metrics)
                
                # Print results
                print(f"\nEpoch {epoch}:")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Val   - F1 (macro): {val_metrics['f1_macro']:.4f}, MAE: {val_metrics['mae']:.4f}")
                
                if self.config.enable_continual_metrics:
                    print(f"  Val   - Forward Transfer: {val_metrics.get('forward_transfer', 0):.4f}")
                    print(f"  Val   - Early Acc: {val_metrics.get('early_accuracy', 0):.4f}, Late Acc: {val_metrics.get('late_accuracy', 0):.4f}")
                
                # Save best model
                if val_metrics['accuracy'] > self.results['best_val_acc']:
                    self.results['best_val_acc'] = val_metrics['accuracy']
                    if self.config.training.save_checkpoint:
                        checkpoint_path = os.path.join(
                            self.config.training.checkpoint_dir,
                            f"{self.config.name}_best.pt"
                        )
                        save_checkpoint(
                            self.model, self.optimizer, epoch,
                            val_metrics, checkpoint_path
                        )
        
        # Save final results
        results_path = os.path.join("results", f"{self.config.name}_results.json")
        os.makedirs("results", exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        
        with open(results_path, 'w') as f:
            json.dump(convert_to_serializable(self.results), f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        print(f"Best validation accuracy: {self.results['best_val_acc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train BDH sentiment analysis model")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples for warm-up")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--max_iters', type=int, default=500, help="Maximum iterations")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--name', type=str, default="bdh_sentiment_warmup", help="Experiment name")
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        name=args.name,
        data=DataConfig(
            num_samples=args.num_samples,
            batch_size=args.batch_size
        ),
        training=TrainingConfig(
            learning_rate=args.learning_rate,
            max_iters=args.max_iters
        )
    )
    
    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
