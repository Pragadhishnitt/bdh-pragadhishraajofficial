"""Utility functions for training and evaluation."""

import torch
import random
import numpy as np
from contextlib import nullcontext


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_and_dtype(device_str: str = "cuda", dtype_str: str = "float16"):
    """
    Get PyTorch device and dtype with automatic fallback.
    
    Args:
        device_str: "cuda", "cpu", or "mps"
        dtype_str: "float32", "float16", or "bfloat16"
        
    Returns:
        device, dtype, context manager, scaler
    """
    # Device selection
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        dtype_str = "float32"  # Force float32 on CPU
    
    # Dtype selection
    if dtype_str == "bfloat16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
    elif dtype_str == "float16" and device.type == "cuda":
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32
        dtype_str = "float32"
    
    # Context manager for mixed precision
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=ptdtype)
        if device.type == "cuda"
        else nullcontext()
    )
    
    # Gradient scaler for float16
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=(dtype_str == "float16")
    )
    
    print(f"Using device: {device} with dtype: {dtype_str}")
    
    # Enable TF32 for better performance on Ampere GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return device, ptdtype, ctx, scaler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    filepath: str
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})
