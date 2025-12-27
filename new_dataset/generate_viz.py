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

import numpy as np
import torch
import bdh
from config import get_default_config
from dragon_metrics import compute_tmi, export_topology

try:
    from visualization import plot_topology, plot_sps_curve, plot_sec_scatter
    VIZ_AVAILABLE = True
except ImportError as e:
    VIZ_AVAILABLE = False
    print(f"Warning: visualization module not available. Error: {e}")


def generate_visualizations(checkpoint_path: str, output_dir: str = "output", skip_topology: bool = False):
    """
    Generate all visualizations from a trained checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for plots
        skip_topology: Skip slow topology/TMI computation (saves 5-10 mins)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("VISUALIZATION GENERATION")
    print("="*70)
    
    # ========================================================================
    # STEP 1: FAST PLOTS FROM SAVED JSON (Instant, <5 seconds)
    # ========================================================================
    
    results_path = os.path.join(output_dir, "hebbian_results.json")
    if os.path.exists(results_path):
        print(f"\n[1/4] Loading hebbian evaluation results...")
        print(f"      File: {results_path}")
        
        with open(results_path) as f:
            results = json.load(f)
        
        print(f"      ✓ Trained BDH PPL: {results.get('trained_bdh_mean_perplexity', 'N/A'):.2f}")
        print(f"      ✓ SPS Hurst: {results.get('sps_hurst_exponent', 'N/A'):.4f}")
        
        # Plot SPS curve (FAST - uses saved data)
        if VIZ_AVAILABLE and 'sps_snapshots' in results:
            print(f"\n[2/4] Plotting SPS curve... ", end='', flush=True)
            try:
                snapshots = [np.array(s) for s in results['sps_snapshots']]
                plot_sps_curve(
                    snapshots, 
                    results.get('sps_hurst_exponent', 0.5),
                    output_path=os.path.join(output_dir, "sps_curve.png")
                )
            except Exception as e:
                print(f"\n      ✗ Failed: {e}")
        
        # Plot SEC scatter (FAST - uses saved data)
        if VIZ_AVAILABLE and 'sec_sparsity_values' in results and 'sec_perplexity_values' in results:
            print(f"\n[3/4] Plotting SEC scatter... ", end='', flush=True)
            try:
                plot_sec_scatter(
                    results['sec_sparsity_values'],
                    results['sec_perplexity_values'],
                    results.get('sec_correlation', 0.0),
                    results.get('sec_p_value', 1.0),
                    output_path=os.path.join(output_dir, "sec_scatter.png")
                )
            except Exception as e:
                print(f"\n      ✗ Failed: {e}")
    else:
        print(f"\n[1/4] ⚠ No hebbian_results.json found at: {results_path}")
        print(f"      Run evaluate_hebbian.py first to generate SPS/SEC plots")
    
    # ========================================================================
    # STEP 2: TOPOLOGY/TMI PLOTS (SLOW - 5-10 minutes, skip with --skip-topology)
    # ========================================================================
    
    if skip_topology:
        print(f"\n[4/4] ⊘ Skipping topology/TMI (use --no-skip-topology to enable)")
        print(f"      Reason: TMI computes 32768×32768 matrix (~5-10 mins)")
    else:
        print(f"\n[4/4] Computing topology/TMI...")
        print(f"      ⚠ WARNING: This is SLOW (5-10 minutes)")
        print(f"      Computing 32768×32768 adjacency matrix...")
        print(f"      (Use --skip-topology flag to skip this)")
        
        # Load model
        config = get_default_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n      Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
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
        
        # Compute TMI
        print(f"\n      Step 1/3: Computing TMI (this takes time)...")
        try:
            tmi_result = compute_tmi(model, top_k_percent=2.0)
            print(f"      ✓ TMI Q={tmi_result.modularity_q:.4f}, Communities={tmi_result.num_communities}")
        except Exception as e:
            print(f"      ✗ TMI failed: {e}")
            tmi_result = None
        
        # Generate topology plot
        if VIZ_AVAILABLE and tmi_result:
            print(f"\n      Step 2/3: Generating topology plot...")
            try:
                plot_topology(model, output_path=os.path.join(output_dir, "topology.svg"), max_nodes=300)
            except Exception as e:
                print(f"      ✗ Topology plot failed: {e}")
        
        # Export GEXF
        print(f"\n      Step 3/3: Exporting GEXF file...")
        try:
            gexf_path = export_topology(model, epoch=9999, output_dir=output_dir, max_nodes=300)
        except Exception as e:
            print(f"      ✗ GEXF export failed: {e}")
    
    print("\n" + "="*70)
    print("✓ Visualization generation complete!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast mode (skip slow topology):
  python generate_viz.py --checkpoint model.pt --skip-topology
  
  # Full mode (includes topology, ~10 mins):
  python generate_viz.py --checkpoint model.pt
"""
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--skip-topology", action="store_true", 
                       help="Skip slow topology/TMI computation (saves 5-10 mins)", default=False)
    args = parser.parse_args()
    
    generate_visualizations(args.checkpoint, args.output_dir, args.skip_topology)


if __name__ == "__main__":
    main()
