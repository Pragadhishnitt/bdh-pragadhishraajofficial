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


def get_config(mode: str, sector: str = "all"):
    """Get configuration based on mode and sector."""
    from config import (
        get_default_config, get_small_config, get_debug_config, get_a100_config,
        get_tech_config, get_tech_quick_config,
    )
    
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
        config.training.log_freq = 1000        # Log every 1000 steps
        config.training.eval_freq = 1000       # Eval every 1000 steps  
        config.training.save_freq = 2500       # Save checkpoints at 2500 & 5000
    elif mode == "a100":
        # A100 optimized (80-90GB VRAM)
        config = get_a100_config()
        print("A100 Mode: batch_size=128, n_layer=8, n_embd=512, bfloat16")
    elif mode == "tech":
        # Technology sector focused - 12k training, 1500 eval
        config = get_tech_config()
        print(f"Tech Mode: 12k training iters, 1500 eval steps, tech filter")
    elif mode == "tech_quick":
        # Quick tech test
        config = get_tech_quick_config()
        print("Tech Quick Mode: 100 iters for testing")
    elif mode == "eval" and sector == "technology":
        # If evaluating tech sector, use tech config
        config = get_tech_config()
        print(f"Eval Mode (Tech): Using tech config (1500 eval steps)")
    else:
        config = get_default_config()
        # Override default eval steps to 1500
        config.metrics.eval_max_steps = 1501
    
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


def find_checkpoint():
    """Find the best available checkpoint."""
    # Priority order
    candidates = [
        "outputs/checkpoints/model_final.pt",
        "weights/model_final.pt",
        "../weights/model_final.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def run_pipeline(mode: str = "full", sector: str = "all", reg: str = "both"):
    """
    Run the complete pipeline.
    
    Args:
        mode: Pipeline mode (full/train/eval/viz/quick/medium/a100/tech/tech_quick)
        sector: Sector filter (technology/healthcare/financials/energy/all)
        reg: Regularization type (l1/l21/both)
    
    Modes:
        full   - Train + Eval + Viz (complete pipeline)
        train  - Train only (Stage A)
        eval   - Eval + Viz (will auto-train if no checkpoint)
        viz    - Visualization only (requires checkpoint)
        quick  - Quick test (100 iters, for debugging)
        medium - Balanced for Kaggle (3000 iters)
        a100   - A100 GPU optimized (batch=128, 10K iters, bfloat16)
    """
    print("\n" + "=" * 60)
    print(f"BDH COMPETITION PIPELINE - MODE: {mode.upper()}")
    if sector != "all":
        print(f"SECTOR: {sector.upper()}")
    print(f"REGULARIZATION: {reg.upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Setup
    dirs = setup_directories()
    check_environment()
    config = get_config(mode, sector)
    
    # Apply sector filter (if specified)
    if sector != "all":
        config.data.filter_strategy = sector
        print(f"\n✓ Data filter: {sector} sector")
    
    # Apply regularization type
    if reg == "l1":
        # L1 only: disable L2,1
        config.training.l21_lambda = 0.0
        print(f"✓ Regularization: L1 only (λ={config.training.l1_lambda})")
    elif reg == "l21":
        # L2,1 only: disable L1
        config.training.l1_lambda = 0.0
        print(f"✓ Regularization: L2,1 only (λ={config.training.l21_lambda})")
    else:  # both
        print(f"✓ Regularization: L1 + L2,1 (λ_l1={config.training.l1_lambda}, λ_l21={config.training.l21_lambda})")
    model = None
    checkpoint_path = find_checkpoint()
    
    # Determine if we need to train
    need_training = mode in ["full", "train", "quick", "medium", "tech", "tech_quick"]
    
    # For eval/a100 mode: auto-train if no checkpoint exists
    if mode in ["eval", "a100"] and checkpoint_path is None:
        print("\n⚠ No checkpoint found. Will train first, then evaluate.")
        need_training = True
    
    # For viz mode: require checkpoint
    if mode == "viz" and checkpoint_path is None:
        print("ERROR: --mode viz requires an existing checkpoint.")
        print("Run with --mode eval or --mode train first.")
        return
    
    # Stage A: Training
    if need_training:
        model = run_training(config)
        checkpoint_path = os.path.join(config.output_dir, "model_final.pt")
    
    # Load model if not already loaded (for eval/viz only modes)
    if model is None and checkpoint_path:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        import bdh
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # Use config from checkpoint (matches training)
            cfg = checkpoint.get("config", None)
            if cfg is None:
                from config import get_small_config
                cfg = get_small_config()
            
            bdh_config = bdh.BDHConfig(
                n_layer=cfg.model.n_layer,
                n_embd=cfg.model.n_embd,
                n_head=cfg.model.n_head,
                dropout=cfg.model.dropout,
                mlp_internal_dim_multiplier=cfg.model.mlp_internal_dim_multiplier,
                vocab_size=cfg.model.vocab_size,
            )
            model = bdh.BDH(bdh_config)
            
            # Fix for torch.compile adding _orig_mod prefix
            state_dict = checkpoint["model_state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            print("✓ Model loaded")
            
        except Exception as e:
            print(f"\n✗ ERROR: Corrupted checkpoint at {checkpoint_path}")
            print(f"  Details: {e}")
            
            if mode in ["a100", "eval"]:
                print("  ⚠ Falling back to training (auto-recovery)...")
                model = run_training(config)
                # Update checkpoint path to the new one
                checkpoint_path = os.path.join(config.output_dir, "model_final.pt")
            else:
                print("  Cannot auto-recover in this mode. Please delete the corrupted file.")
                raise e
    
    # Stage B: Evaluation
    if mode in ["full", "eval", "a100"]:
        if checkpoint_path and os.path.exists(checkpoint_path):
            run_evaluation(checkpoint_path, config)
        else:
            print("Skipping evaluation - no checkpoint found")
    
    # Stage C: Visualization
    if mode in ["full", "eval", "viz", "quick", "a100"]:
        if model is not None:
            run_visualization(model, config)
        else:
            print("Skipping visualization - no model available")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  Checkpoints:     outputs/checkpoints/")
    print("  Metrics:         outputs/metrics/")
    print("  Visualizations:  outputs/visualizations/")
    print("\nGenerated files:")
    for file in ["outputs/visualizations/topology_final.svg", 
                 "outputs/metrics/hebbian_results.json",
                 "outputs/metrics/tmi_results.json"]:
        if os.path.exists(file):
            print(f"  ✓ {file}")
    print("\nNext steps:")
    print("  1. Review outputs/visualizations/topology_final.svg")
    print("  2. Check outputs/metrics/hebbian_results.json")
    print("  3. Upload outputs/visualizations/topology_final.gexf to Gephi")


def main():
    parser = argparse.ArgumentParser(
        description="BDH Competition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full       - Complete pipeline: Train + Eval + Viz (8000 iters)
  train      - Training only (Stage A)
  eval       - Evaluation + Viz (auto-trains if no checkpoint)
  viz        - Visualization only (requires checkpoint)
  quick      - Quick test (100 iters, for debugging)
  medium     - Balanced for Kaggle (3000 iters)
  a100       - A100 optimized mode
  tech       - Tech sector preset (12k train + 6k eval)
  tech_quick - Tech sector quick test

Sectors:
  technology  - Tech companies (FAANG+, semiconductors, software)
  healthcare  - Pharma, providers, medical devices
  financials  - Banks, asset management, insurance
  energy      - Oil & gas exploration, refining, services
  all         - No filtering (default)

Examples:
  python pipeline.py --mode train --sector technology --reg l1
  python pipeline.py --mode train --sector healthcare --reg l21
  python pipeline.py --mode train --sector financials --reg both
  python pipeline.py --mode eval --sector technology
"""
    )
    parser.add_argument(
        "--mode", 
        choices=["full", "medium", "train", "eval", "viz", "quick", "a100", 
                 "tech", "tech_quick"],
        default="eval",
        help="Pipeline mode (default: eval)"
    )
    parser.add_argument(
        "--sector",
        choices=["technology", "healthcare", "financials", "energy", "all"],
        default="all",
        help="Sector to train on (default: all)"
    )
    parser.add_argument(
        "--reg",
        choices=["l1", "l21", "both"],
        default="both",
        help="Regularization type for Stage A: l1 (activation), l21 (weight clustering), both (default: both)"
    )
    args = parser.parse_args()
    
    run_pipeline(args.mode, sector=args.sector, reg=args.reg)


if __name__ == "__main__":
    main()
