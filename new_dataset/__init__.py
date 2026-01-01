"""
new_dataset package initialization

This package contains the BDH competition implementation modules.
"""

from .config import (
    ExperimentConfig, get_default_config, get_small_config, get_debug_config,
    get_tech_config, get_tech_refine_config, get_config, list_configs,
)
from .finance_loader import FinanceDataset, create_data_loaders
from .dragon_metrics import compute_tmi, compute_sps, compute_sec, compute_mcl
from .train_finance import train
from .evaluate_hebbian import evaluate_hebbian
from .visualization import plot_topology, plot_sparsity_heatmap, plot_activation_atlas
from .data_filter import get_filter_strategy, TechnologyFilter, AllDataFilter
from .sector_registry import get_sector, get_symbols_for_sector, list_sectors

__all__ = [
    # Config
    "ExperimentConfig",
    "get_default_config",
    "get_small_config", 
    "get_debug_config",
    "get_tech_config",
    "get_tech_refine_config",
    "get_config",
    "list_configs",
    # Data
    "FinanceDataset",
    "create_data_loaders",
    "get_filter_strategy",
    "TechnologyFilter",
    "AllDataFilter",
    "get_sector",
    "get_symbols_for_sector",
    "list_sectors",
    # Metrics
    "compute_tmi",
    "compute_sps",
    "compute_sec",
    "compute_mcl",
    # Training
    "train",
    "evaluate_hebbian",
    # Visualization
    "plot_topology",
    "plot_sparsity_heatmap",
    "plot_activation_atlas",
]

