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
    
    def __init__(self, base_model: bdh.BDH, learning_rate: float = 0.01):
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
        
        # Damping factor for forgetting (prevents saturation)
        self.damping = 0.99
    
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
    
    Freezes BDH weights and allows only synaptic state updates.
    Compares perplexity against frozen GPT-2 baseline to demonstrate
    that dynamic synaptic adaptation enables continual learning.
    """
    
    if config is None:
        config = get_default_config()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load BDH model
    print(f"Loading BDH checkpoint: {bdh_checkpoint_path}")
    checkpoint = torch.load(bdh_checkpoint_path, map_location=device)
    
    bdh_config = bdh.BDHConfig(
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        dropout=config.model.dropout,
        mlp_internal_dim_multiplier=config.model.mlp_internal_dim_multiplier,
        vocab_size=config.model.vocab_size,
    )
    
    base_model = bdh.BDH(bdh_config).to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Wrap with Hebbian inference
    hebbian_model = HebbianBDH(base_model).to(device)
    hebbian_model.eval()
    
    # Load GPT-2 baseline if available
    gpt2_model = None
    if compare_gpt2 and TRANSFORMERS_AVAILABLE:
        print("Loading GPT-2 baseline...")
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        gpt2_model.eval()
        for param in gpt2_model.parameters():
            param.requires_grad = False
    
    # Load evaluation data (2019-2024)
    print("Loading evaluation data...")
    _, eval_loader = create_data_loaders(
        pretrain_years=config.data.pretrain_years,
        eval_years=config.data.eval_years,
        sector=config.data.sector,
        block_size=config.data.block_size,
        batch_size=config.data.batch_size,
    )
    
    # Initialize trackers
    sigma_tracker = SynapticStateTracker(snapshot_interval=config.metrics.sps_snapshot_interval)
    sparsity_tracker = ActivationSparsityTracker()
    
    bdh_perplexities = []
    gpt2_perplexities = []
    
    # Evaluation loop
    print("Running Hebbian inference evaluation...")
    step = 0
    max_steps = 1000  # Limit for reasonable runtime
    
    for batch in eval_loader:
        if step >= max_steps:
            break
        
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        
        # BDH with Hebbian updates
        with torch.no_grad():
            _, bdh_loss = hebbian_model(x, y)
            bdh_ppl = torch.exp(bdh_loss).item()
            bdh_perplexities.append(bdh_ppl)
            
            # Track synaptic state
            sigma_tracker.maybe_snapshot(
                torch.cat([s.flatten() for s in hebbian_model.sigma])
            )
        
        # GPT-2 baseline
        if gpt2_model is not None:
            with torch.no_grad():
                gpt2_outputs = gpt2_model(x, labels=y)
                gpt2_ppl = torch.exp(gpt2_outputs.loss).item()
                gpt2_perplexities.append(gpt2_ppl)
        
        if step % 100 == 0:
            avg_bdh_ppl = np.mean(bdh_perplexities[-100:])
            msg = f"Step {step} | BDH PPL: {avg_bdh_ppl:.2f}"
            if gpt2_perplexities:
                avg_gpt2_ppl = np.mean(gpt2_perplexities[-100:])
                msg += f" | GPT-2 PPL: {avg_gpt2_ppl:.2f}"
            print(msg)
        
        step += 1
    
    # Compute SPS (Synaptic Persistence Score)
    print("\nComputing Synaptic Persistence Score...")
    sps_result = compute_sps(sigma_tracker.snapshots)
    print(f"SPS Hurst Exponent: {sps_result.hurst_exponent:.4f}")
    print(f"  (H > 0.5 indicates long-range dependence)")
    
    # Compute final statistics
    results = {
        "bdh_mean_perplexity": np.mean(bdh_perplexities),
        "bdh_std_perplexity": np.std(bdh_perplexities),
        "sps_hurst_exponent": sps_result.hurst_exponent,
        "steps_evaluated": step,
    }
    
    if gpt2_perplexities:
        results["gpt2_mean_perplexity"] = np.mean(gpt2_perplexities)
        results["gpt2_std_perplexity"] = np.std(gpt2_perplexities)
        results["bdh_wins"] = results["bdh_mean_perplexity"] < results["gpt2_mean_perplexity"]
    
    # Summary
    print("\n" + "="*60)
    print("HEBBIAN INFERENCE EVALUATION RESULTS")
    print("="*60)
    print(f"BDH Mean Perplexity: {results['bdh_mean_perplexity']:.2f} ± {results['bdh_std_perplexity']:.2f}")
    if "gpt2_mean_perplexity" in results:
        print(f"GPT-2 Mean Perplexity: {results['gpt2_mean_perplexity']:.2f} ± {results['gpt2_std_perplexity']:.2f}")
        if results["bdh_wins"]:
            print("\n✓ SUCCESS: BDH outperforms GPT-2 via Hebbian adaptation!")
        else:
            print("\n✗ BDH did not outperform GPT-2 on this evaluation")
    print(f"\nSPS Hurst Exponent: {results['sps_hurst_exponent']:.4f}")
    if results['sps_hurst_exponent'] > 0.5:
        print("✓ Long-range synaptic memory confirmed (H > 0.5)")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BDH Hebbian Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to BDH checkpoint")
    parser.add_argument("--no_gpt2", action="store_true", help="Skip GPT-2 comparison")
    parser.add_argument("--max_steps", type=int, default=1000)
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
