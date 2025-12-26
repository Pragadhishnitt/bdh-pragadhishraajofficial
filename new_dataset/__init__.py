"""
new_dataset package initialization

This package contains the BDH competition implementation modules.
"""

from .config import ExperimentConfig, get_default_config, get_small_config, get_debug_config
from .finance_loader import FinanceDataset, create_data_loaders
from .dragon_metrics import compute_tmi, compute_sps, compute_sec, compute_mcl
from .train_finance import train
from .evaluate_hebbian import evaluate_hebbian
from .visualization import plot_topology, plot_sparsity_heatmap, plot_activation_atlas

__all__ = [
    "ExperimentConfig",
    "get_default_config",
    "get_small_config", 
    "get_debug_config",
    "FinanceDataset",
    "create_data_loaders",
    "compute_tmi",
    "compute_sps",
    "compute_sec",
    "compute_mcl",
    "train",
    "evaluate_hebbian",
    "plot_topology",
    "plot_sparsity_heatmap",
    "plot_activation_atlas",
]
