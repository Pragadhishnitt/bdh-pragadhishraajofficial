"""
BDH training script for financial earnings transcripts.

Two-stage protocol:
1. Pre-training (2005-2018) with L1 sparsity regularization
2. Hebbian inference evaluation (2019-2024) - see evaluate_hebbian.py
"""

import os
import argparse
from contextlib import nullcontext
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import bdh
from config import ExperimentConfig, get_default_config, get_small_config, get_debug_config
from finance_loader import FinanceDataset, create_data_loaders
from dragon_metrics import (
    compute_tmi, export_topology,
    ActivationSparsityTracker, compute_sec,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ActivationHookManager:
    """
    Manages forward hooks to capture sparse activations during training.
    
    Intercepts post-ReLU activations without disrupting the forward pass.
    Used for computing L1 sparsity penalty and activation statistics.
    """
    
    def __init__(self):
        self.activations: List[torch.Tensor] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
    
    def register_hooks(self, model: nn.Module) -> None:
        """Register hooks on BDH model to capture post-ReLU activations."""
        # Clear any existing hooks
        self.remove_hooks()
        self.activations = []
        
        # Hook the forward method to capture x_sparse
        # We'll use a wrapper approach since BDH doesn't have explicit ReLU modules
        original_forward = model.forward
        
        def hooked_forward(idx, targets=None):
            self.activations = []  # Reset for this forward pass
            
            # Run original forward
            C = model.config
            B, T = idx.size()
            D = C.n_embd
            nh = C.n_head
            N = D * C.mlp_internal_dim_multiplier // nh
            
            x = model.embed(idx).unsqueeze(1)
            x = model.ln(x)
            
            for level in range(C.n_layer):
                x_latent = x @ model.encoder
                x_sparse = F.relu(x_latent)
                
                # Capture sparse activations
                self.activations.append(x_sparse.detach())
                
                yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
                yKV = model.ln(yKV)
                
                y_latent = yKV @ model.encoder_v
                y_sparse = F.relu(y_latent)
                xy_sparse = x_sparse * y_sparse
                xy_sparse = model.drop(xy_sparse)
                
                yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder)
                y = model.ln(yMLP)
                x = model.ln(x + y)
            
            logits = x.view(B, T, D) @ model.lm_head
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            return logits, loss
        
        # Store hook handle (for compatibility)
        self._original_forward = original_forward
        model.forward = hooked_forward
    
    def remove_hooks(self) -> None:
        """Remove registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def get_sparsity_penalty(self) -> torch.Tensor:
        """
        Compute L1 penalty on activations to encourage sparsity.
        
        Returns average L1 norm across all captured activations.
        """
        if not self.activations:
            return torch.tensor(0.0)
        
        total_penalty = 0.0
        for act in self.activations:
            total_penalty = total_penalty + torch.mean(torch.abs(act))
        
        return total_penalty / len(self.activations)
    
    def get_mean_sparsity(self) -> float:
        """Get mean activation density (fraction non-zero)."""
        if not self.activations:
            return 0.0
        
        densities = [(act > 0).float().mean().item() for act in self.activations]
        return sum(densities) / len(densities)


def setup_training(config: ExperimentConfig) -> Tuple:
    """Initialize model, optimizer, and training context."""
    
    # Device setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dtype setup
    dtype = config.dtype
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(dtype, torch.float32)
    
    # Autocast context
    if "cuda" in str(device):
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
    else:
        ctx = nullcontext()
        scaler = None
    
    # Model setup
    bdh_config = bdh.BDHConfig(
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        dropout=config.model.dropout,
        mlp_internal_dim_multiplier=config.model.mlp_internal_dim_multiplier,
        vocab_size=config.model.vocab_size,
    )
    
    model = bdh.BDH(bdh_config).to(device)
    
    # Try to compile (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("Model compiled successfully")
    except Exception as e:
        print(f"Model compilation not available: {e}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Learning rate scheduler (cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.max_iters,
    )
    
    return model, optimizer, scheduler, scaler, ctx, device


def train(config: Optional[ExperimentConfig] = None, dry_run: bool = False):
    """
    Main BDH training loop with L1 sparsity regularization.
    
    Trains on 2005-2018 earnings transcripts using:
    - Autoregressive language modeling objective
    - AdamW optimizer with cosine LR schedule  
    - L1 penalty on activations for sparsity
    """
    
    if config is None:
        config = get_default_config()
    
    # Initialize WandB
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.project_name,
            config={
                "model": config.model.__dict__,
                "training": config.training.__dict__,
                "data": config.data.__dict__,
            },
        )
    
    # Setup
    model, optimizer, scheduler, scaler, ctx, device = setup_training(config)
    hook_manager = ActivationHookManager()
    hook_manager.register_hooks(model)
    
    sparsity_tracker = ActivationSparsityTracker()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Data loader (now with train/val/test split)
    print("Loading training data...")
    train_loader, val_loader, eval_loader = create_data_loaders(
        pretrain_years=config.data.pretrain_years,
        eval_years=config.data.eval_years,
        sector=config.data.sector,
        block_size=config.data.block_size,
        batch_size=config.data.batch_size,
    )
    
    # Training loop with graceful handling
    print(f"Starting training for {config.training.max_iters} iterations...")
    
    model.train()
    step = 0
    loss_acc = 0.0
    loss_steps = 0
    data_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    # TMI checkpoints: midpoint and end only
    tmi_steps = {config.training.max_iters // 2, config.training.max_iters - 1}
    
    def save_checkpoint(step, reason="scheduled"):
        """Save checkpoint with error handling."""
        checkpoint_path = os.path.join(config.output_dir, f"checkpoint_{step}.pt")
        try:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)
            print(f"  Saved checkpoint ({reason}): {checkpoint_path}")
            return True
        except Exception as e:
            print(f"  WARNING: Checkpoint save failed: {e}")
            return False
    
    def gpu_memory_check():
        """Check GPU memory and cooldown if needed."""
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            usage_pct = mem_used / mem_total
            
            if usage_pct > 0.85:  # Over 85% usage
                print(f"  GPU memory high ({usage_pct*100:.1f}%), cooling down...")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                import time
                time.sleep(5)
                print("  Resuming training...")
    
    try:
        for step in range(config.training.max_iters):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            
            # Forward pass with autocast
            with ctx:
                logits, loss = model(x, y)
                
                # L1 sparsity regularization
                sparsity_penalty = hook_manager.get_sparsity_penalty()
                if isinstance(sparsity_penalty, torch.Tensor):
                    sparsity_penalty = sparsity_penalty.to(device)
                total_loss = loss + config.training.l1_lambda * sparsity_penalty
            
            # Track metrics
            loss_acc += loss.item()
            loss_steps += 1
            
            mean_sparsity = hook_manager.get_mean_sparsity()
            sparsity_tracker.record(
                activations=hook_manager.activations[-1] if hook_manager.activations else torch.zeros(1),
                loss=loss.item(),
            )
            
            # Backward pass
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            
            # Logging
            if step % config.training.log_freq == 0:
                avg_loss = loss_acc / loss_steps
                current_lr = scheduler.get_last_lr()[0]
                
                print(f"Step {step}/{config.training.max_iters} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Sparsity: {mean_sparsity:.4f} | "
                      f"LR: {current_lr:.6f}")
                
                if config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/sparsity": mean_sparsity,
                        "train/lr": current_lr,
                        "train/l1_penalty": sparsity_penalty.item() if isinstance(sparsity_penalty, torch.Tensor) else sparsity_penalty,
                    }, step=step)
                
                loss_acc = 0.0
                loss_steps = 0
            
            # Validation (every eval_freq, lightweight - no heavy metrics)
            if step % config.training.eval_freq == 0 and step > 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for _ in range(5):
                        try:
                            val_batch = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            val_batch = next(val_iter)
                        
                        val_x = val_batch["input_ids"].to(device)
                        val_y = val_batch["labels"].to(device)
                        _, val_loss = model(val_x, val_y)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"  Val Loss: {avg_val_loss:.4f}")
                
                if config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({"val/loss": avg_val_loss}, step=step)
                
                model.train()
                
                # GPU memory check during validation
                gpu_memory_check()
            
            # TMI only at midpoint and end (expensive computation)
            if step in tmi_steps:
                print(f"\n  Computing TMI at step {step}...")
                try:
                    tmi_result = compute_tmi(model, top_k_percent=config.metrics.tmi_top_k_percent)
                    print(f"  TMI: Q={tmi_result.modularity_q:.4f}, "
                          f"Communities={tmi_result.num_communities}")
                    
                    if config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "metrics/tmi_modularity": tmi_result.modularity_q,
                            "metrics/tmi_communities": tmi_result.num_communities,
                        }, step=step)
                except Exception as e:
                    print(f"  TMI computation failed: {e}")
                
                # Cooldown after heavy computation
                gpu_memory_check()
            
            # Save checkpoint
            if step % config.training.save_freq == 0 and step > 0:
                save_checkpoint(step)
                
                # Export topology
                try:
                    gexf_path = export_topology(model, step, config.output_dir)
                    print(f"  Exported topology: {gexf_path}")
                except Exception as e:
                    print(f"  Topology export failed: {e}")
            
            if dry_run and step >= 10:
                print("Dry run complete!")
                break
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("TRAINING INTERRUPTED - Saving checkpoint...")
        print(f"{'='*60}")
        save_checkpoint(step, reason="interrupted")
        print("You can resume from this checkpoint later.")
        print(f"{'='*60}\n")
    
    # Final save
    final_path = os.path.join(config.output_dir, "model_final.pt")
    try:
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": config,
        }, final_path)
        print(f"Training complete! Final model saved to: {final_path}")
    except Exception as e:
        print(f"Final save failed: {e}")
    
    # Final TMI (at the very end)
    print("\nComputing final TMI...")
    try:
        tmi_result = compute_tmi(model)
        print(f"Final TMI: Q={tmi_result.modularity_q:.4f}, Communities={tmi_result.num_communities}")
    except Exception as e:
        print(f"Final TMI computation failed: {e}")
    
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train BDH on Financial Data")
    parser.add_argument("--config", choices=["default", "small", "debug"], default="default")
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    # Get config
    if args.config == "default":
        config = get_default_config()
    elif args.config == "small":
        config = get_small_config()
    else:
        config = get_debug_config()
    
    # Override settings
    if args.max_iters:
        config.training.max_iters = args.max_iters
    if args.no_wandb:
        config.use_wandb = False
    
    train(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
