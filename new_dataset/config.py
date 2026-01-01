"""
BDH Competition Experiment Configuration

Centralized hyperparameters matching specifications.
Open/Closed Principle: Add new configs without modifying existing ones.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class ModelConfig:
    """BDH model architecture configuration."""
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    dropout: float = 0.1
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 50257  # GPT-2 BPE tokenizer vocab size


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    dataset_name: str = "kurry/sp500_earnings_transcripts"
    filter_strategy: str = "all"  # "all", "technology", "sector:<name>"
    pretrain_years: Tuple[int, int] = (2005, 2018)  # Stage A
    eval_years: Tuple[int, int] = (2019, 2024)       # Stage B
    block_size: int = 512
    batch_size: int = 32


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    max_iters: int = 8000
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    l1_lambda: float = 2e-3  # Sparsity regularization (increased for better modularity)
    log_freq: int = 100
    eval_freq: int = 500
    save_freq: int = 1000


@dataclass
class MetricsConfig:
    """Frontier metrics configuration."""
    tmi_top_k_percent: float = 2.3  # Edge sparsification threshold
    sps_snapshot_interval: int = 100  # Tokens between σ snapshots
    eval_max_steps: int = 3000  # Max steps for Hebbian evaluation (Stage B)
    concepts: List[str] = field(default_factory=lambda: [
        "Inflation", "AI", "Dividend", "Layoffs", "Revenue", 
        "Guidance", "Headwinds", "EBITDA", "Margin"
    ])


@dataclass
class ThresholdAnalysisConfig:
    """Configuration for sparsity threshold analysis."""
    enabled: bool = False
    sparsity_thresholds: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.3, 0.5, 0.7, 0.9
    ])
    output_file: str = "threshold_analysis.json"


@dataclass
class ExperimentConfig:
    """Master configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    threshold_analysis: ThresholdAnalysisConfig = field(default_factory=ThresholdAnalysisConfig)
    
    # Experiment tracking
    use_wandb: bool = True
    project_name: str = "bdh-finance"
    output_dir: str = "output"
    
    # Device settings
    device: str = "cuda"
    dtype: str = "bfloat16"


# =============================================================================
# CONFIG PRESETS (Open for Extension)
# =============================================================================

def get_default_config() -> ExperimentConfig:
    """Return default experiment configuration."""
    return ExperimentConfig()


def get_small_config() -> ExperimentConfig:
    """Smaller config for limited compute (e.g., T4 GPU)."""
    config = ExperimentConfig()
    config.model.n_layer = 4
    config.model.n_embd = 128
    config.data.batch_size = 16
    config.training.max_iters = 8000
    return config


def get_debug_config() -> ExperimentConfig:
    """Minimal config for debugging."""
    config = ExperimentConfig()
    config.model.n_layer = 2
    config.model.n_embd = 64
    config.data.batch_size = 4
    config.data.block_size = 128
    config.training.max_iters = 100
    config.training.log_freq = 10
    config.use_wandb = False
    return config


def get_a100_config() -> ExperimentConfig:
    """
    Optimized config for A100 GPU (40GB VRAM).
    
    Uses moderate model size that fits in memory.
    """
    config = ExperimentConfig()
    
    # Moderate model - fits in 40GB A100
    config.model.n_layer = 6          # Reduced from 8
    config.model.n_embd = 256         # Reduced from 512
    config.model.n_head = 4           # Reduced from 8
    config.model.mlp_internal_dim_multiplier = 128
    
    # Conservative batch size
    config.data.batch_size = 32
    config.data.block_size = 512
    
    # Full training
    config.training.max_iters = 8000
    config.training.log_freq = 500
    config.training.eval_freq = 1000
    config.training.save_freq = 1000
    
    # A100 native bfloat16
    config.dtype = "bfloat16"
    
    return config


# =============================================================================
# TECHNOLOGY-FOCUSED PRESET
# =============================================================================

def get_tech_config() -> ExperimentConfig:
    """
    Technology sector focused pipeline for P100/Kaggle/Colab.
    
    Uses small config (batch=32) for 16GB GPUs.
    - Filters data to Technology sector only
    - Stage A (Training): 12k iterations
    - Stage B (Hebbian Eval): 6k steps
    """
    config = get_small_config()  # Use small config for P100 (16GB)
    
    # Technology filter - only difference in data
    config.data.filter_strategy = "technology"
    
    # Stage A: 12k training iterations
    config.training.max_iters = 12000
    config.training.log_freq = 1000
    config.training.eval_freq = 2000
    config.training.save_freq = 4000
    
    # Stage B: 6k evaluation steps
    config.metrics.eval_max_steps = 6000
    
    return config


def get_tech_quick_config() -> ExperimentConfig:
    """Quick tech test config (100 training iters, 50 eval steps)."""
    config = get_tech_config()
    config.training.max_iters = 100
    config.training.log_freq = 10
    config.training.eval_freq = 50
    config.training.save_freq = 50
    config.metrics.eval_max_steps = 50
    config.use_wandb = False
    return config


# =============================================================================
# CONFIG FACTORY
# =============================================================================

_CONFIG_REGISTRY = {
    "default": get_default_config,
    "small": get_small_config,
    "debug": get_debug_config,
    "a100": get_a100_config,
    "tech": get_tech_config,
    "tech_quick": get_tech_quick_config,
}


def get_config(name: str) -> ExperimentConfig:
    """
    Factory function to get config by name.
    
    Args:
        name: Config preset name
        
    Returns:
        ExperimentConfig instance
    """
    name_lower = name.lower()
    if name_lower not in _CONFIG_REGISTRY:
        available = list(_CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config: '{name}'. Available: {available}")
    return _CONFIG_REGISTRY[name_lower]()


def list_configs() -> List[str]:
    """List all available config presets."""
    return list(_CONFIG_REGISTRY.keys())


