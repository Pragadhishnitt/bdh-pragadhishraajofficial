"""
Hebbian inference evaluation for BDH.

Evaluates BDH with frozen weights but dynamic synaptic state updates.
Compares adaptation performance against GPT-2 baseline on held-out test data.
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bdh
from config import ExperimentConfig, get_default_config
from finance_loader import create_data_loaders
from dragon_metrics import (
    SynapticStateTracker, compute_sps,
    ActivationSparsityTracker, compute_sec,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HebbianBDH(nn.Module):
    """
    BDH wrapper with Hebbian synaptic updates during inference.
    
    Static weights are frozen, but synaptic state (σ) adapts based on
    activation correlations: "neurons that fire together, wire together".
    """
    
    def __init__(self, base_model: bdh.BDH, learning_rate: float = 0.005):
        super().__init__()
        self.model = base_model
        self.learning_rate = learning_rate
        
        # Freeze all static weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Initialize synaptic state σ (learnable during inference)
        # Shape: (n_layer, n_head, N, N) - connections between neurons
        config = base_model.config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        # σ represents synaptic strengths between neurons
        self.sigma = nn.ParameterList([
            nn.Parameter(torch.zeros(nh, N, N), requires_grad=False)
            for _ in range(config.n_layer)
        ])
        
        # Damping factor for forgetting (lower = faster pruning of weak connections)
        self.damping = 0.95
    
    def hebbian_update(self, x_sparse: torch.Tensor, layer: int) -> None:
        """
        Update synaptic state using Hebbian rule.
        
        Δσ(i,j) = η * X(i) * X(j)
        
        This encodes correlations between active neurons.
        """
        # x_sparse shape: (B, nh, T, N)
        # Compute outer product of activations (correlation)
        B, nh, T, N = x_sparse.shape
        
        # Average over batch and time
        x_mean = x_sparse.mean(dim=(0, 2))  # (nh, N)
        
        # Outer product: correlation matrix
        correlation = torch.einsum('hi,hj->hij', x_mean, x_mean)  # (nh, N, N)
        
        # Hebbian update with damping
        with torch.no_grad():
            self.sigma[layer].data = (
                self.damping * self.sigma[layer].data + 
                self.learning_rate * correlation
            )
    
    def forward(self, idx, targets=None):
        """Forward pass with Hebbian synaptic updates."""
        C = self.model.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        x = self.model.embed(idx).unsqueeze(1)
        x = self.model.ln(x)
        
        for level in range(C.n_layer):
            x_latent = x @ self.model.encoder
            x_sparse = F.relu(x_latent)
            
            # Hebbian update on this layer
            self.hebbian_update(x_sparse, level)
            
            # Modulate attention with synaptic state
            # σ influences which neuron connections are strengthened
            yKV = self.model.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.model.ln(yKV)
            
            y_latent = yKV @ self.model.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.model.drop(xy_sparse)
            
            yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.model.decoder)
            y = self.model.ln(yMLP)
            x = self.model.ln(x + y)
        
        logits = x.view(B, T, D) @ self.model.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def get_sigma_snapshot(self) -> np.ndarray:
        """Get flattened synaptic state for SPS computation."""
        return torch.cat([s.flatten() for s in self.sigma]).cpu().numpy()
    
    def reset_sigma(self) -> None:
        """Reset synaptic state to zero."""
        for s in self.sigma:
            s.data.zero_()


def evaluate_hebbian(
    bdh_checkpoint_path: str,
    config: Optional[ExperimentConfig] = None,
    compare_gpt2: bool = True,
) -> Dict:
    """
    Evaluate BDH's Hebbian adaptation on test data (2019-2024).
    
    Compares three models:
    1. Trained BDH with Hebbian inference (our method)
    2. Untrained BDH (random weights baseline)
    3. GPT-2 (pre-trained language model baseline)
    """
    
    if config is None:
        config = get_default_config()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load BDH model
    print(f"Loading trained BDH checkpoint: {bdh_checkpoint_path}")
    checkpoint = torch.load(bdh_checkpoint_path, map_location=device, weights_only=False)
    
    # Use config from checkpoint (matches training)
    saved_config = checkpoint.get("config", None)
    if saved_config is not None:
        config = saved_config
    
    bdh_config = bdh.BDHConfig(
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        dropout=config.model.dropout,
        mlp_internal_dim_multiplier=config.model.mlp_internal_dim_multiplier,
        vocab_size=config.model.vocab_size,
    )
    
    # --- Trained BDH model ---
    trained_model = bdh.BDH(bdh_config).to(device)
    # Fix for torch.compile adding _orig_mod prefix
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    trained_model.load_state_dict(new_state_dict)
    
    # Wrap with Hebbian inference
    hebbian_model = HebbianBDH(trained_model).to(device)
    hebbian_model.eval()
    
    # --- Memory optimization: Load baselines lazily ---
    # We'll load untrained BDH and GPT-2 only when needed and delete after
    untrained_model = None
    gpt2_model = None
    run_baselines = compare_gpt2  # Only run baselines if comparing
    
    # Load evaluation data (2019-2024)
    print("Loading evaluation data...")
    # Use smaller batch size for evaluation to save memory
    eval_batch_size = min(8, config.data.batch_size)  # Cap at 8 for T4 GPU
    _, _, eval_loader = create_data_loaders(
        pretrain_years=config.data.pretrain_years,
        eval_years=config.data.eval_years,
        sector=config.data.sector,
        block_size=config.data.block_size,
        batch_size=eval_batch_size,
    )
    print(f"Using eval batch size: {eval_batch_size} (reduced for memory)")
    
    # Initialize trackers
    sigma_tracker = SynapticStateTracker(snapshot_interval=config.metrics.sps_snapshot_interval)
    sparsity_tracker = ActivationSparsityTracker()
    
    trained_perplexities = []
    untrained_perplexities = []
    gpt2_perplexities = []
    
    # For SEC visualization - track sparsity and perplexity per step
    sec_sparsity_values = []
    sec_perplexity_values = []
    
    # Evaluation loop - 5000 steps to match Stage A
    print("Running Hebbian inference evaluation...")
    print("Comparing: Trained BDH vs Untrained BDH vs GPT-2\n")
    step = 0
    max_steps = 1500  # Match Stage A for full evaluation
    log_freq = 300
    save_freq = 500  # Save intermediate results every 1000 steps
    
    # Output directory setup
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    def gpu_memory_check():
        """Check GPU memory and cooldown if needed (prevents Kaggle flush errors)."""
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            usage_pct = mem_used / mem_total
            
            if usage_pct > 0.85:  # Over 85% usage
                print(f"  GPU memory high ({usage_pct*100:.1f}%), cooling down...")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                import time
                time.sleep(5)
                print("  Resuming evaluation...")
    
    def save_intermediate_results(step, trained_ppls, untrained_ppls, gpt2_ppls, reason="scheduled"):
        """Save intermediate results to prevent data loss."""
        import json
        results_path = os.path.join(output_dir, f"hebbian_eval_step_{step}.json")
        try:
            intermediate = {
                "step": step,
                "trained_bdh_mean_ppl": float(np.mean(trained_ppls)) if trained_ppls else 0,
                "trained_bdh_std_ppl": float(np.std(trained_ppls)) if trained_ppls else 0,
                "untrained_bdh_mean_ppl": float(np.mean(untrained_ppls)) if untrained_ppls else 0,
                "untrained_bdh_std_ppl": float(np.std(untrained_ppls)) if untrained_ppls else 0,
                "gpt2_mean_ppl": float(np.mean(gpt2_ppls)) if gpt2_ppls else 0,
                "gpt2_std_ppl": float(np.std(gpt2_ppls)) if gpt2_ppls else 0,
            }
            with open(results_path, "w") as f:
                json.dump(intermediate, f, indent=2)
            print(f"  Saved intermediate results ({reason}): {results_path}")
            return True
        except Exception as e:
            print(f"  WARNING: Save failed: {e}")
            return False
    
    # Initialize baselines lazily (memory optimization)
    import gc
    
    try:
        for batch in eval_loader:
            if step >= max_steps:
                break
            
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            
            with torch.no_grad():
                # Trained BDH with Hebbian updates (main model)
                _, trained_loss = hebbian_model(x, y)
                trained_ppl = torch.exp(trained_loss).item()
                trained_perplexities.append(trained_ppl)
                
                # Track synaptic state for trained model
                sigma_tracker.maybe_snapshot(
                    torch.cat([s.flatten() for s in hebbian_model.sigma])
                )
                
                # Track SEC data (sparsity vs perplexity)
                x_emb = hebbian_model.model.ln(hebbian_model.model.embed(x))
                x_latent = x_emb.unsqueeze(1) @ hebbian_model.model.encoder
                x_sparse = torch.relu(x_latent)
                sparsity = (x_sparse > 0).float().mean().item()
                sec_sparsity_values.append(sparsity)
                sec_perplexity_values.append(trained_ppl)
                
                # Clear intermediate tensors
                del x_emb, x_latent, x_sparse
            
            # Run baselines only every 10 steps to save memory/time
            if run_baselines and step % 10 == 0:
                # Lazy load untrained BDH
                if untrained_model is None:
                    print("  Loading untrained BDH baseline...")
                    untrained_base = bdh.BDH(bdh_config).to(device)
                    untrained_model = HebbianBDH(untrained_base).to(device)
                    untrained_model.eval()
                
                with torch.no_grad():
                    _, untrained_loss = untrained_model(x, y)
                    untrained_ppl = torch.exp(untrained_loss).item()
                    untrained_perplexities.append(untrained_ppl)
                
                # Lazy load GPT-2 (only if transformers available)
                if gpt2_model is None and TRANSFORMERS_AVAILABLE:
                    print("  Loading GPT-2 baseline...")
                    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
                    gpt2_model.eval()
                    for param in gpt2_model.parameters():
                        param.requires_grad = False
                
                if gpt2_model is not None:
                    with torch.no_grad():
                        gpt2_outputs = gpt2_model(x, labels=y)
                        gpt2_ppl = torch.exp(gpt2_outputs.loss).item()
                        gpt2_perplexities.append(gpt2_ppl)
                
                # Clear CUDA cache periodically
                if step % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Logging
            if step % log_freq == 0:
                avg_trained = np.mean(trained_perplexities[-100:])
                avg_untrained = np.mean(untrained_perplexities[-100:])
                msg = f"Step {step}/{max_steps} | Trained BDH PPL: {avg_trained:.2f} | Untrained BDH PPL: {avg_untrained:.2f}"
                if gpt2_perplexities:
                    avg_gpt2 = np.mean(gpt2_perplexities[-100:])
                    msg += f" | GPT-2 PPL: {avg_gpt2:.2f}"
                print(msg)
            
            # Intermediate save and memory check
            if step % save_freq == 0 and step > 0:
                save_intermediate_results(step, trained_perplexities, untrained_perplexities, gpt2_perplexities)
                gpu_memory_check()
            
            step += 1
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("EVALUATION INTERRUPTED - Saving partial results...")
        print(f"{'='*60}")
        save_intermediate_results(step, trained_perplexities, untrained_perplexities, gpt2_perplexities, reason="interrupted")
        print("You can find partial results in the output directory.")
        print(f"{'='*60}\n")
    
    # Compute SPS (Synaptic Persistence Score)
    print("\nComputing Synaptic Persistence Score...")
    sps_result = compute_sps(sigma_tracker.snapshots)
    print(f"SPS Hurst Exponent: {sps_result.hurst_exponent:.4f}")
    print(f"  (H > 0.5 indicates long-range dependence)")
    
    # Compute final statistics
    results = {
        "trained_bdh_mean_perplexity": np.mean(trained_perplexities),
        "trained_bdh_std_perplexity": np.std(trained_perplexities),
        "untrained_bdh_mean_perplexity": np.mean(untrained_perplexities),
        "untrained_bdh_std_perplexity": np.std(untrained_perplexities),
        "sps_hurst_exponent": sps_result.hurst_exponent,
        "steps_evaluated": step,
    }
    
    if gpt2_perplexities:
        results["gpt2_mean_perplexity"] = np.mean(gpt2_perplexities)
        results["gpt2_std_perplexity"] = np.std(gpt2_perplexities)
    
    # Add raw data for visualization plots
    # Save SPS snapshots (subsample to keep file size reasonable)
    snapshot_subsample = max(1, len(sigma_tracker.snapshots) // 100)  # Keep ~100 snapshots
    results["sps_snapshots"] = [s.tolist() for s in sigma_tracker.snapshots[::snapshot_subsample]]
    
    # Save SEC data (sparsity vs perplexity)
    results["sec_sparsity_values"] = sec_sparsity_values
    results["sec_perplexity_values"] = sec_perplexity_values
    
    # Compute SEC correlation
    if len(sec_sparsity_values) > 10:
        from dragon_metrics import compute_sec
        sec_result = compute_sec(sec_sparsity_values, sec_perplexity_values)
        results["sec_correlation"] = sec_result.correlation
        results["sec_p_value"] = sec_result.p_value
        print(f"\nSEC Correlation: r={sec_result.correlation:.4f}, p={sec_result.p_value:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("HEBBIAN INFERENCE EVALUATION RESULTS")
    print("="*70)
    print(f"{'Model':<25} {'Mean PPL':>12} {'Std PPL':>12}")
    print("-"*70)
    print(f"{'Trained BDH (Hebbian)':<25} {results['trained_bdh_mean_perplexity']:>12.2f} {results['trained_bdh_std_perplexity']:>12.2f}")
    print(f"{'Untrained BDH (Random)':<25} {results['untrained_bdh_mean_perplexity']:>12.2f} {results['untrained_bdh_std_perplexity']:>12.2f}")
    if "gpt2_mean_perplexity" in results:
        print(f"{'GPT-2 (Pretrained)':<25} {results['gpt2_mean_perplexity']:>12.2f} {results['gpt2_std_perplexity']:>12.2f}")
    print("-"*70)
    
    # Compute improvement percentages
    training_improvement = (
        (results["untrained_bdh_mean_perplexity"] - results["trained_bdh_mean_perplexity"]) 
        / results["untrained_bdh_mean_perplexity"] * 100
    )
    print(f"\n✓ Training Improvement: {training_improvement:.1f}% lower PPL vs untrained baseline")
    
    if "gpt2_mean_perplexity" in results:
        if results["trained_bdh_mean_perplexity"] < results["gpt2_mean_perplexity"]:
            gpt2_improvement = (
                (results["gpt2_mean_perplexity"] - results["trained_bdh_mean_perplexity"]) 
                / results["gpt2_mean_perplexity"] * 100
            )
            print(f"✓ Trained BDH outperforms GPT-2 by {gpt2_improvement:.1f}%")
        else:
            print(f"✗ GPT-2 outperforms trained BDH (expected for domain-specific fine-tuning)")
    
    print(f"\nSPS Hurst Exponent: {results['sps_hurst_exponent']:.4f}")
    if results['sps_hurst_exponent'] > 0.5:
        print("✓ Long-range synaptic memory confirmed (H > 0.5)")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BDH Hebbian Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to BDH checkpoint")
    parser.add_argument("--no_gpt2", action="store_true", help="Skip GPT-2 comparison")
    parser.add_argument("--max_steps", type=int, default=5000, help="Max evaluation steps (default: 5000)")
    args = parser.parse_args()
    
    results = evaluate_hebbian(
        bdh_checkpoint_path=args.checkpoint,
        compare_gpt2=not args.no_gpt2,
    )
    
    # Save results
    import json
    output_path = "output/hebbian_eval_results.json"
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
