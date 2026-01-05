"""
Quick verification script to check model parameter count.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models import BDHSentiment, BDHSentimentConfig

# Create model with default config
config = BDHSentimentConfig()

model = BDHSentiment(config)

# Count parameters
total_params = model.count_parameters()
print(f"\n{'='*50}")
print(f"BDH Sentiment Model Parameter Count")
print(f"{'='*50}")
print(f"Total parameters: {total_params:,}")
print(f"In millions: {total_params/1e6:.2f}M")
print(f"{'='*50}")

if total_params <= 30e6:
    print(f"✓ Model is within the 30M parameter limit")
else:
    print(f"✗ Model exceeds the 30M parameter limit by {(total_params - 30e6)/1e6:.2f}M")

print(f"\nModel configuration:")
print(f"  Layers: {config.n_layer}")
print(f"  Embedding dim: {config.n_embd}")
print(f"  Attention heads: {config.n_head}")
print(f"  MLP multiplier: {config.mlp_internal_dim_multiplier}")
print(f"  Vocab size: {config.vocab_size}")
print(f"  Num classes: {config.num_classes}")
