"""
Data loading and processing module for sentiment analysis.
Preserves temporal ordering for continual learning evaluation.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Tuple, Optional, List
import numpy as np


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis with temporal ordering.
    
    CRITICAL: Does NOT shuffle data to preserve temporal order for
    continual learning evaluation.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        timestamps: List[str],
        tokenizer,
        max_length: int = 128
    ):
        """
        Args:
            texts: List of tweet texts
            labels: List of sentiment labels (0-4)
            timestamps: List of timestamps (for tracking temporal order)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.timestamps = timestamps
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]
        timestamp = self.timestamps[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'timestamp': timestamp,
            'text': text  # Keep for error analysis
        }


class SentimentDataLoader:
    """
    Data loader that maintains temporal ordering.
    Handles sequential train/val split without shuffling.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_seq_length: int = 128,
        num_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        batch_size: int = 8
    ):
        """
        Args:
            data_path: Path to processed_agi_sentiment.csv
            tokenizer_name: Name of HuggingFace tokenizer
            max_seq_length: Maximum sequence length
            num_samples: Number of samples to use (None = all)
            train_ratio: Ratio for sequential train/val split
            batch_size: Batch size for DataLoader
        """
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        
        self._load_data()
        
    def _load_data(self):
        """Load and prepare data while preserving temporal order."""
        # Load CSV
        df = pd.read_csv(self.data_path)
        
        # Sort by time to ensure chronological order
        df = df.sort_values('time').reset_index(drop=True)
        
        # Limit samples if specified (for warm-up run)
        if self.num_samples is not None:
            df = df.head(self.num_samples)
        
        # Sequential split (NO SHUFFLE)
        split_idx = int(len(df) * self.train_ratio)
        
        self.train_df = df.iloc[:split_idx].reset_index(drop=True)
        self.val_df = df.iloc[split_idx:].reset_index(drop=True)
        
        print(f"Data loaded: {len(df)} total samples")
        print(f"  Train: {len(self.train_df)} samples (temporal range: {self.train_df['time'].min()} to {self.train_df['time'].max()})")
        print(f"  Val: {len(self.val_df)} samples (temporal range: {self.val_df['time'].min()} to {self.val_df['time'].max()})")
        
    def get_datasets(self) -> Tuple[SentimentDataset, SentimentDataset]:
        """Get train and validation datasets."""
        train_dataset = SentimentDataset(
            texts=self.train_df['text'].tolist(),
            labels=self.train_df['label'].tolist(),
            timestamps=self.train_df['time'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length
        )
        
        val_dataset = SentimentDataset(
            texts=self.val_df['text'].tolist(),
            labels=self.val_df['label'].tolist(),
            timestamps=self.val_df['time'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length
        )
        
        return train_dataset, val_dataset
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation DataLoaders.
        
        CRITICAL: shuffle=False to preserve temporal ordering.
        """
        train_dataset, val_dataset = self.get_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # CRITICAL: No shuffling
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # CRITICAL: No shuffling
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_full_dataframe(self) -> pd.DataFrame:
        """Get the full dataframe (for analysis)."""
        return pd.concat([self.train_df, self.val_df], ignore_index=True)
