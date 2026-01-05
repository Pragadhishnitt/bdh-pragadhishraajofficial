"""
Visualization module for sentiment analysis results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd


class ResultsVisualizer:
    """Visualize training results and continual learning metrics."""
    
    def __init__(self, results_path: str, output_dir: str = "results/figures"):
        """
        Args:
            results_path: Path to results JSON file
            output_dir: Directory to save figures
        """
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    
    def plot_training_curves(self):
        """Plot loss and accuracy curves over training."""
        train_history = self.results['train_history']
        val_history = self.results['val_history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        train_losses = [h['loss'] for h in train_history]
        val_losses = [h['loss'] for h in val_history]
        
        ax1.plot(train_losses, label='Train', linewidth=2)
        ax1.plot(np.linspace(0, len(train_losses), len(val_losses)), val_losses, 
                label='Validation', linewidth=2, marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        train_accs = [h['accuracy'] for h in train_history]
        val_accs = [h['accuracy'] for h in val_history]
        
        ax2.plot(train_accs, label='Train', linewidth=2)
        ax2.plot(np.linspace(0, len(train_accs), len(val_accs)), val_accs,
                label='Validation', linewidth=2, marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'training_curves.png'}")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix."""
        if not self.results['val_history']:
            return
        
        cm = np.array(self.results['val_history'][-1]['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Very Neg', 'Mild Neg', 'Neutral', 'Mild Pos', 'Very Pos'],
                   yticklabels=['Very Neg', 'Mild Neg', 'Neutral', 'Mild Pos', 'Very Pos'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Validation Set)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'confusion_matrix.png'}")
    
    def plot_per_class_metrics(self):
        """Plot per-class precision, recall, and F1."""
        if not self.results['val_history']:
            return
        
        last_val = self.results['val_history'][-1]
        
        classes = ['Very Neg', 'Mild Neg', 'Neutral', 'Mild Pos', 'Very Pos']
        precision = [last_val[f'precision_class_{i}'] for i in range(5)]
        recall = [last_val[f'recall_class_{i}'] for i in range(5)]
        f1 = [last_val[f'f1_class_{i}'] for i in range(5)]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Sentiment Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics (Validation Set)')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'per_class_metrics.png'}")
    
    def plot_temporal_analysis(self):
        """Plot temporal accuracy and concept drift."""
        if not self.results['val_history'] or 'temporal_accuracies' not in self.results['val_history'][-1]:
            return
        
        last_val = self.results['val_history'][-1]
        temporal_accs = last_val['temporal_accuracies']
        window_indices = last_val['window_indices']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Temporal accuracy
        ax1.plot(window_indices, temporal_accs, marker='o', linewidth=2, markersize=8)
        ax1.axhline(y=np.mean(temporal_accs), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(temporal_accs):.3f}')
        ax1.set_xlabel('Sample Index (Time Window Start)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Temporal Accuracy (Rolling Window)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution shift
        early_dist = np.array(last_val['early_distribution'])
        late_dist = np.array(last_val['late_distribution'])
        
        x = np.arange(5)
        width = 0.35
        
        ax2.bar(x - width/2, early_dist, width, label='Early Period', alpha=0.8)
        ax2.bar(x + width/2, late_dist, width, label='Late Period', alpha=0.8)
        ax2.set_xlabel('Sentiment Class')
        ax2.set_ylabel('Proportion')
        ax2.set_title('Concept Drift: Prediction Distribution Shift')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Very Neg', 'Mild Neg', 'Neutral', 'Mild Pos', 'Very Pos'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'temporal_analysis.png'}")
    
    def plot_continual_learning_metrics(self):
        """Plot continual learning specific metrics."""
        if not self.results['val_history']:
            return
        
        last_val = self.results['val_history'][-1]
        
        if 'forward_transfer' not in last_val:
            return
        
        metrics = {
            'Forward Transfer': last_val.get('forward_transfer', 0),
            'Early Accuracy': last_val.get('early_accuracy', 0),
            'Late Accuracy': last_val.get('late_accuracy', 0),
            'Distribution Shift': last_val.get('distribution_shift', 0)
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in metrics.values()]
        bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=colors, alpha=0.7)
        
        ax.set_xlabel('Value')
        ax.set_title('Continual Learning Metrics')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (k, v) in enumerate(metrics.items()):
            ax.text(v, i, f' {v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'continual_learning_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'continual_learning_metrics.png'}")
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\nGenerating visualizations...")
        self.plot_training_curves()
        self.plot_confusion_matrix()
        self.plot_per_class_metrics()
        self.plot_temporal_analysis()
        self.plot_continual_learning_metrics()
        print(f"\nAll plots saved to: {self.output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize sentiment analysis results")
    parser.add_argument('--results', type=str, required=True, help="Path to results JSON file")
    parser.add_argument('--output_dir', type=str, default="results/figures", help="Output directory")
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.results, args.output_dir)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
