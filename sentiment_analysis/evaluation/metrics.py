"""
Evaluation metrics for sentiment analysis.
Includes standard classification metrics and continual learning metrics.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score, mean_absolute_error
)
from typing import List, Dict, Tuple, Optional
import pandas as pd


class MetricsCalculator:
    """
    Calculate comprehensive metrics for sentiment analysis.
    Includes both standard and continual learning metrics.
    """
    
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.all_preds = []
        self.all_labels = []
        self.all_timestamps = []
        
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        timestamps: Optional[List[str]] = None
    ):
        """
        Update with new batch of predictions.
        
        Args:
            predictions: (batch_size, num_classes) logits or (batch_size,) class indices
            labels: (batch_size,) ground truth labels
            timestamps: List of timestamps for temporal analysis
        """
        if predictions.dim() == 2:
            # Convert logits to class predictions
            preds = predictions.argmax(dim=-1)
        else:
            preds = predictions
        
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        
        if timestamps is not None:
            self.all_timestamps.extend(timestamps)
    
    def compute_standard_metrics(self) -> Dict[str, float]:
        """
        Compute standard classification metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, F1, etc.
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Overall accuracy
        acc = accuracy_score(labels, preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
        
        # Sentiment-specific metrics
        mae = mean_absolute_error(labels, preds)  # Treating labels as ordinal
        qwk = cohen_kappa_score(labels, preds, weights='quadratic')
        
        metrics = {
            'accuracy': acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'mae': mae,
            'quadratic_kappa': qwk,
            'confusion_matrix': cm.tolist()
        }
        
        # Add per-class metrics
        for i in range(self.num_classes):
            metrics[f'precision_class_{i}'] = precision[i] if i < len(precision) else 0.0
            metrics[f'recall_class_{i}'] = recall[i] if i < len(recall) else 0.0
            metrics[f'f1_class_{i}'] = f1[i] if i < len(f1) else 0.0
            metrics[f'support_class_{i}'] = int(support[i]) if i < len(support) else 0
        
        return metrics
    
    def compute_continual_learning_metrics(
        self,
        window_size: int = 10
    ) -> Dict[str, any]:
        """
        Compute continual learning metrics.
        
        Args:
            window_size: Size of rolling window for temporal analysis
            
        Returns:
            Dictionary with continual learning metrics
        """
        if len(self.all_timestamps) == 0:
            return {}
        
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        n_samples = len(preds)
        
        # Handle very small datasets
        if n_samples < 5:
            return {
                'temporal_accuracies': [],
                'window_indices': [],
                'early_accuracy': 0.0,
                'late_accuracy': 0.0,
                'forward_transfer': 0.0,
                'distribution_shift': 0.0,
                'early_distribution': [0] * self.num_classes,
                'late_distribution': [0] * self.num_classes
            }
        
        # Temporal accuracy (accuracy over time windows)
        temporal_accs = []
        window_indices = []
        
        for i in range(0, n_samples, window_size):
            end_idx = min(i + window_size, n_samples)
            if end_idx - i < 2:  # Skip if window too small
                continue
            window_preds = preds[i:end_idx]
            window_labels = labels[i:end_idx]
            window_acc = accuracy_score(window_labels, window_preds)
            temporal_accs.append(window_acc)
            window_indices.append(i)
        
        # Forward transfer: compare early vs late performance
        # Early = first 20%, Late = last 20%
        split_idx = max(1, n_samples // 5)  # At least 1 sample
        
        # Ensure we have enough samples for split
        if split_idx >= n_samples:
            split_idx = max(1, n_samples // 2)
        
        early_acc = accuracy_score(labels[:split_idx], preds[:split_idx]) if split_idx > 0 else 0
        late_acc = accuracy_score(labels[-split_idx:], preds[-split_idx:]) if split_idx > 0 else 0
        forward_transfer = late_acc - early_acc
        
        # Concept drift: distribution shift in predictions
        early_dist = np.bincount(preds[:split_idx], minlength=self.num_classes) / max(split_idx, 1)
        late_dist = np.bincount(preds[-split_idx:], minlength=self.num_classes) / max(split_idx, 1)
        distribution_shift = np.linalg.norm(early_dist - late_dist)
        
        metrics = {
            'temporal_accuracies': temporal_accs,
            'window_indices': window_indices,
            'early_accuracy': early_acc,
            'late_accuracy': late_acc,
            'forward_transfer': forward_transfer,
            'distribution_shift': distribution_shift,
            'early_distribution': early_dist.tolist(),
            'late_distribution': late_dist.tolist()
        }
        
        return metrics
    
    def compute_all_metrics(self, window_size: int = 10) -> Dict[str, any]:
        """Compute all metrics (standard + continual learning)."""
        standard = self.compute_standard_metrics()
        continual = self.compute_continual_learning_metrics(window_size)
        
        return {**standard, **continual}
    
    def get_predictions_dataframe(self) -> pd.DataFrame:
        """Get predictions as a DataFrame for analysis."""
        return pd.DataFrame({
            'prediction': self.all_preds,
            'label': self.all_labels,
            'timestamp': self.all_timestamps if self.all_timestamps else [None] * len(self.all_preds),
            'correct': np.array(self.all_preds) == np.array(self.all_labels)
        })
