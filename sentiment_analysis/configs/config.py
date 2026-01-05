"""
Configuration module for sentiment analysis experiments.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for BDH model architecture."""
    n_layer: int = 4
    n_embd: int = 384
    n_head: int = 6
    mlp_internal_dim_multiplier: int = 64
    vocab_size: int = 30522  # DistilBERT vocab size
    dropout: float = 0.2
    num_classes: int = 5  # Sentiment classes: 0-4
    max_seq_length: int = 128


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    data_path: str = "processed_agi_sentiment.csv"  # Relative to project root
    tokenizer_name: str = "distilbert-base-uncased"
    max_seq_length: int = 128
    num_samples: Optional[int] = 100  # For warm-up run
    train_ratio: float = 0.8  # Sequential split
    preserve_temporal_order: bool = True  # CRITICAL: No shuffling
    batch_size: int = 8


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_iters: int = 500
    eval_interval: int = 50
    log_interval: int = 10
    save_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"  # Will auto-detect
    dtype: str = "float16"  # or "bfloat16" or "float32"
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "bdh_sentiment_warmup"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Continual learning evaluation
    enable_continual_metrics: bool = True
    time_window_size: int = 10  # For rolling window analysis
    
    # Baseline comparison
    compare_with_baseline: bool = True
    baseline_model: str = "distilbert"  # or "lstm"
