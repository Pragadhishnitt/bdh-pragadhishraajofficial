"""
BDH Competition Pipeline - Kaggle/Colab Ready

Usage:
    !python pipeline.py --mode full        # Full pipeline (5000 iters)
    !python pipeline.py --mode medium      # Balanced for Kaggle (2000 iters)
    !python pipeline.py --mode train       # Training only
    !python pipeline.py --mode eval        # Evaluation only
    !python pipeline.py --mode quick       # Quick test (100 iters)
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add new_dataset to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "new_dataset"))

import torch


def setup_directories():
    """Create output directory structure."""
    dirs = [
        "outputs",
        "outputs/checkpoints",
        "outputs/metrics",
        "outputs/visualizations",
        "outputs/logs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return {name.split("/")[-1]: name for name in dirs}


def check_environment():
    """Check and report environment setup."""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("✗ GPU: Not available, using CPU (will be slow)")
    
    # Dependencies
    deps = {
        "networkx": "Graph analysis (TMI metric)",
        "community": "Community detection (Louvain)",
        "wandb": "Experiment tracking",
        "datasets": "HuggingFace data loading",
        "tiktoken": "GPT-2 tokenization",
        "matplotlib": "Visualization",
        "transformers": "GPT-2 baseline comparison",
    }
    
    for pkg, purpose in deps.items():
        try:
            __import__(pkg)
            print(f"✓ {pkg}: {purpose}")
        except ImportError:
            print(f"✗ {pkg}: {purpose} (optional, fallback available)")
    
    print("=" * 60)


def get_config(mode: str):
    """Get configuration based on mode."""
    from config import get_default_config, get_small_config, get_debug_config
    
    if mode == "quick":
        config = get_debug_config()
        config.training.max_iters = 100
        config.training.log_freq = 10
        config.training.eval_freq = 50
        config.training.save_freq = 50
    elif mode == "medium":
        # Balanced mode for Kaggle - reduces I/O to avoid timeouts
        config = get_small_config()
        config.training.max_iters = 3000       # vs 5000 in full
        config.training.log_freq = 600         # Less printing
        config.training.eval_freq = 1000        # Less frequent validation
        config.training.save_freq = 1500       # Save only at 1000 & 2000
    elif mode in ["full", "train"]:
        # Use small config for Kaggle/Colab (T4 GPU)
        config = get_small_config()
    else:
        config = get_default_config()
    
    # Ensure outputs go to our directory
    config.output_dir = "outputs/checkpoints"
    config.use_wandb = False  # Disable for simplicity in notebooks
    
    return config


def run_training(config):
    """Run Stage A: Structural Pre-training."""
    print("\n" + "=" * 60)
    print("STAGE A: STRUCTURAL PRE-TRAINING")
    print("=" * 60)
    
    from train_finance import train
    model = train(config)
    
    return model


def run_evaluation(checkpoint_path: str, config):
    """Run Stage B: Hebbian Inference Evaluation."""
    print("\n" + "=" * 60)
    print("STAGE B: HEBBIAN INFERENCE EVALUATION")
    print("=" * 60)
    
    from evaluate_hebbian import evaluate_hebbian
    results = evaluate_hebbian(
        checkpoint_path,
        config,
        compare_gpt2=True,
    )
    
    # Save results
    results_path = "outputs/metrics/hebbian_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    return results


def run_visualization(model, config):
    """Generate all visualizations."""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    try:
        from visualization import (
            plot_topology, 
            plot_metrics_dashboard,
            export_topology_gexf,
        )
        from dragon_metrics import compute_tmi
        
        # Topology plot
        print("Generating topology plot...")
        plot_topology(
            model, 
            output_path="outputs/visualizations/topology_final.svg",
        )
        
        # Export GEXF for Gephi
        print("Exporting topology to GEXF...")
        export_topology_gexf(
            model,
            output_path="outputs/visualizations/topology_final.gexf",
        )
        
        # Compute final TMI
        print("Computing final TMI...")
        tmi = compute_tmi(model)
        print(f"  Modularity Q = {tmi.modularity_q:.4f}")
        print(f"  Communities = {tmi.num_communities}")
        
        # Save TMI results
        tmi_results = {
            "modularity_q": tmi.modularity_q,
            "num_communities": tmi.num_communities,
            "community_sizes": tmi.community_sizes[:10],  # Top 10
        }
        with open("outputs/metrics/tmi_results.json", "w") as f:
            json.dump(tmi_results, f, indent=2)
        
        print("Visualizations saved to outputs/visualizations/")
        
    except Exception as e:
        print(f"Visualization error (non-fatal): {e}")


def run_pipeline(mode: str = "full"):
    """Run the complete pipeline."""
    print("\n" + "=" * 60)
    print(f"BDH COMPETITION PIPELINE - MODE: {mode.upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Setup
    dirs = setup_directories()
    check_environment()
    config = get_config(mode)
    
    # Training
    if mode in ["full", "train", "quick", "medium"]:
        model = run_training(config)
        checkpoint_path = os.path.join(config.output_dir, "model_final.pt")
    else:
        # Eval only - need existing checkpoint
        checkpoint_path = "outputs/checkpoints/model_final.pt"
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            print("Run with --mode train first.")
            return
        
        # Load model for visualization
        import bdh
        from config import get_default_config
        cfg = get_default_config()
        bdh_config = bdh.BDHConfig(
            n_layer=cfg.model.n_layer,
            n_embd=cfg.model.n_embd,
            n_head=cfg.model.n_head,
            vocab_size=cfg.model.vocab_size,
        )
        model = bdh.BDH(bdh_config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluation
    if mode in ["full", "eval"]:
        if os.path.exists(checkpoint_path):
            run_evaluation(checkpoint_path, config)
        else:
            print("Skipping evaluation - no checkpoint found")
    
    # Visualization
    if mode in ["full", "eval", "quick"]:
        run_visualization(model, config)
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  Checkpoints:     outputs/checkpoints/")
    print("  Metrics:         outputs/metrics/")
    print("  Visualizations:  outputs/visualizations/")
    print("\nNext steps:")
    print("  1. Review outputs/visualizations/topology_final.svg")
    print("  2. Check outputs/metrics/hebbian_results.json")
    print("  3. Upload outputs/visualizations/topology_final.gexf to Gephi")


def main():
    parser = argparse.ArgumentParser(description="BDH Competition Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["full", "medium", "train", "eval", "quick"],
        default="medium",
        help="Pipeline mode: full (5000 iters), medium (2000 iters), train, eval, quick (100 iters)"
    )
    args = parser.parse_args()
    
    run_pipeline(args.mode)


if __name__ == "__main__":
    main()
