"""
Standalone visualization generator.

Run this after hebbian evaluation to generate plots without re-running evaluation.
"""

import os
import json
import argparse

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import bdh
from config import get_default_config
from dragon_metrics import compute_tmi, export_topology

try:
    from visualization import (
        plot_topology,
        plot_sps_curve,
        plot_sec_scatter,
    )
    VIZ_AVAILABLE = True
except ImportError as e:
    VIZ_AVAILABLE = False
    print(f"Warning: visualization module not available. Error: {e}")


def generate_visualizations(checkpoint_path: str, output_dir: str = "output"):
    """Generate all visualizations from a trained checkpoint."""
    
    os.makedirs(output_dir, exist_ok=True)
    config = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Rebuild model
    saved_config = checkpoint.get("config", config)
    bdh_config = bdh.BDHConfig(
        n_layer=saved_config.model.n_layer,
        n_embd=saved_config.model.n_embd,
        n_head=saved_config.model.n_head,
        dropout=saved_config.model.dropout,
        mlp_internal_dim_multiplier=saved_config.model.mlp_internal_dim_multiplier,
        vocab_size=saved_config.model.vocab_size,
    )
    
    model = bdh.BDH(bdh_config).to(device)
    
    # Fix torch.compile prefix
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print("\n1. Generating topology plot (optimized, max 500 nodes)...")
    try:
        if VIZ_AVAILABLE:
            from visualization import plot_topology
            plot_topology(model, output_path=os.path.join(output_dir, "topology.svg"), max_nodes=500)
        else:
            print("   Skipped: visualization module not available")
    except Exception as e:
        print(f"   Failed: {e}")
    
    print("\n2. Computing TMI metrics...")
    try:
        tmi_result = compute_tmi(model, top_k_percent=5.0)
        print(f"   TMI Q={tmi_result.modularity_q:.4f}, Communities={tmi_result.num_communities}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    print("\n3. Exporting GEXF topology file...")
    try:
        gexf_path = export_topology(model, epoch=9999, output_dir=output_dir, max_nodes=500)
        print(f"   Saved: {gexf_path}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Load hebbian results if available
    results_path = os.path.join(output_dir, "hebbian_results.json")
    if os.path.exists(results_path):
        print(f"\n4. Loaded existing hebbian results from: {results_path}")
        with open(results_path) as f:
            results = json.load(f)
        print(f"   Trained BDH PPL: {results.get('trained_bdh_mean_perplexity', 'N/A'):.2f}")
        print(f"   SPS Hurst: {results.get('sps_hurst_exponent', 'N/A'):.4f}")
    
    print("\n✓ Visualization generation complete!")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    generate_visualizations(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
