"""
Model Merging for BDH Competition

Implements the "Conglomerate" Hypothesis from PDF Section 7.1:
- Train separate specialist models (Tech-BDH, Pharma-BDH)
- Concatenate neuron graphs: N_total = N_tech + N_pharma
- Merged model competent in both domains without fine-tuning

This demonstrates the "Modular AI paradigm" - training small 
specialized "Hatchlings" and fusing them into a "Dragon".
"""

import os
from typing import List, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn

import bdh
from config import ExperimentConfig, get_default_config


@dataclass
class MergedModelStats:
    """Statistics for the merged model."""
    total_neurons: int
    source_models: List[str]
    neurons_per_model: List[int]
    total_parameters: int


class MergedBDH(nn.Module):
    """
    A merged BDH model created by concatenating specialist models.
    
    From PDF Section 7.1:
    "We create a Merged Model by concatenating their neuron graphs
    (expanding N_total = N_tech + N_pharma) and their static weights."
    
    The merged model should immediately be competent in both domains
    without any fine-tuning.
    """
    
    def __init__(self, source_models: List[bdh.BDH], model_names: Optional[List[str]] = None):
        super().__init__()
        
        if len(source_models) < 2:
            raise ValueError("Need at least 2 models to merge")
        
        self.model_names = model_names or [f"model_{i}" for i in range(len(source_models))]
        self.source_configs = [m.config for m in source_models]
        
        # Verify compatible configurations
        base_config = self.source_configs[0]
        for i, cfg in enumerate(self.source_configs[1:], 1):
            if cfg.n_embd != base_config.n_embd:
                raise ValueError(f"Model {i} has different n_embd: {cfg.n_embd} vs {base_config.n_embd}")
            if cfg.n_layer != base_config.n_layer:
                raise ValueError(f"Model {i} has different n_layer: {cfg.n_layer} vs {base_config.n_layer}")
            if cfg.vocab_size != base_config.vocab_size:
                raise ValueError(f"Model {i} has different vocab_size: {cfg.vocab_size} vs {base_config.vocab_size}")
        
        self.n_layer = base_config.n_layer
        self.n_embd = base_config.n_embd
        self.vocab_size = base_config.vocab_size
        self.n_head = base_config.n_head
        self.dropout = base_config.dropout
        
        # Calculate merged dimensions
        # N per model = mlp_internal_dim_multiplier * n_embd / n_head
        self.n_per_model = [
            cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
            for cfg in self.source_configs
        ]
        self.total_n = sum(n * base_config.n_head for n in self.n_per_model)
        
        # Merge encoders: stack along N dimension
        # encoder shape: (nh, D, N) -> merged: (nh, D, N_total)
        merged_encoders = []
        for m in source_models:
            merged_encoders.append(m.encoder.data)
        self.encoder = nn.Parameter(torch.cat(merged_encoders, dim=2))
        
        # Merge encoder_v similarly
        merged_encoder_v = []
        for m in source_models:
            merged_encoder_v.append(m.encoder_v.data)
        self.encoder_v = nn.Parameter(torch.cat(merged_encoder_v, dim=2))
        
        # Merge decoders: stack along (nh*N) dimension
        # decoder shape: (nh*N, D) -> merged: (nh*N_total, D)
        merged_decoders = []
        for m in source_models:
            merged_decoders.append(m.decoder.data)
        self.decoder = nn.Parameter(torch.cat(merged_decoders, dim=0))
        
        # Use first model's embedding and lm_head (shared vocab)
        self.embed = source_models[0].embed
        self.lm_head = nn.Parameter(source_models[0].lm_head.data.clone())
        
        # Shared modules
        self.ln = nn.LayerNorm(self.n_embd, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(self.dropout)
        
        # Create merged attention
        self.attn = self._create_merged_attention(source_models)
    
    def _create_merged_attention(self, source_models: List[bdh.BDH]) -> nn.Module:
        """Create attention module for merged model."""
        # Use first model's attention but will work with larger N
        base_attn = source_models[0].attn
        
        # Calculate merged N
        nh = self.n_head
        D = self.n_embd
        merged_N = self.encoder.shape[2]  # Total N after concatenation
        
        # Create new attention with merged dimensions
        class MergedAttention(nn.Module):
            def __init__(self, freqs, n_head, merged_n):
                super().__init__()
                # Extend freqs for larger N
                theta = 2**16
                new_freqs = (
                    1.0 / (theta ** (torch.arange(0, merged_n, 1, dtype=torch.float32) / merged_n))
                    / (2 * 3.14159)
                ).view(1, 1, 1, merged_n)
                self.freqs = nn.Buffer(new_freqs)
                self.n_head = n_head
            
            @staticmethod
            def phases_cos_sin(phases):
                import math
                phases = (phases % 1) * (2 * math.pi)
                return torch.cos(phases), torch.sin(phases)
            
            @staticmethod
            def rope(phases, v):
                v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
                phases_cos, phases_sin = MergedAttention.phases_cos_sin(phases)
                return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)
            
            def forward(self, Q, K, V):
                _, _, T, _ = Q.size()
                r_phases = (
                    torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
                    .view(1, 1, -1, 1)
                ) * self.freqs
                QR = self.rope(r_phases, Q)
                KR = QR
                scores = (QR @ KR.mT).tril(diagonal=-1)
                return scores @ V
        
        return MergedAttention(base_attn.freqs, nh, merged_N)
    
    def forward(self, idx, targets=None):
        """Forward pass through merged model."""
        import torch.nn.functional as F
        
        B, T = idx.size()
        D = self.n_embd
        nh = self.n_head
        N = self.encoder.shape[2]  # Merged N
        
        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)
        
        for level in range(self.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            
            xy_sparse = self.drop(xy_sparse)
            
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, nh * N) @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)
        
        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def get_stats(self) -> MergedModelStats:
        """Get statistics about the merged model."""
        total_params = sum(p.numel() for p in self.parameters())
        return MergedModelStats(
            total_neurons=self.encoder.shape[2] * self.n_head,
            source_models=self.model_names,
            neurons_per_model=[n * self.n_head for n in self.n_per_model],
            total_parameters=total_params,
        )


def merge_specialists(
    checkpoint_paths: List[str],
    model_names: Optional[List[str]] = None,
    config: Optional[ExperimentConfig] = None,
) -> MergedBDH:
    """
    Merge multiple specialist BDH models into one.
    
    From PDF Section 7.1:
    "We then create a Merged Model by concatenating their neuron graphs"
    
    Args:
        checkpoint_paths: Paths to specialist model checkpoints
        model_names: Optional names for each model
        config: Experiment configuration
        
    Returns:
        MergedBDH model
    """
    if config is None:
        config = get_default_config()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    models = []
    for path in checkpoint_paths:
        print(f"Loading model from: {path}")
        checkpoint = torch.load(path, map_location=device)
        
        # Get config from checkpoint or use default
        if "config" in checkpoint:
            saved_config = checkpoint["config"]
            bdh_config = bdh.BDHConfig(
                n_layer=saved_config.model.n_layer,
                n_embd=saved_config.model.n_embd,
                n_head=saved_config.model.n_head,
                dropout=saved_config.model.dropout,
                mlp_internal_dim_multiplier=saved_config.model.mlp_internal_dim_multiplier,
                vocab_size=saved_config.model.vocab_size,
            )
        else:
            bdh_config = bdh.BDHConfig(
                n_layer=config.model.n_layer,
                n_embd=config.model.n_embd,
                n_head=config.model.n_head,
                dropout=config.model.dropout,
                mlp_internal_dim_multiplier=config.model.mlp_internal_dim_multiplier,
                vocab_size=config.model.vocab_size,
            )
        
        model = bdh.BDH(bdh_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        models.append(model)
    
    print(f"Merging {len(models)} models...")
    merged = MergedBDH(models, model_names)
    
    stats = merged.get_stats()
    print(f"Merged model stats:")
    print(f"  Total neurons: {stats.total_neurons}")
    print(f"  Source models: {stats.source_models}")
    print(f"  Neurons per model: {stats.neurons_per_model}")
    print(f"  Total parameters: {stats.total_parameters:,}")
    
    return merged.to(device)


def evaluate_merged_model(
    merged_model: MergedBDH,
    domain_data_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate merged model on each domain.
    
    From PDF Section 7.1:
    "Prediction: The merged model should immediately be competent
    in both domains without any fine-tuning."
    """
    results = {}
    merged_model.eval()
    
    for domain_name, loader in domain_data_loaders.items():
        print(f"Evaluating on {domain_name}...")
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                
                _, loss = merged_model(x, y)
                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]
                
                if total_samples >= 1000:  # Limit evaluation
                    break
        
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        results[f"{domain_name}_loss"] = avg_loss
        results[f"{domain_name}_perplexity"] = perplexity
        print(f"  {domain_name}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge BDH Specialist Models")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to checkpoints")
    parser.add_argument("--names", nargs="+", help="Names for each model")
    parser.add_argument("--output", type=str, default="output/merged_model.pt")
    args = parser.parse_args()
    
    merged = merge_specialists(args.checkpoints, args.names)
    
    # Save merged model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "model_state_dict": merged.state_dict(),
        "stats": merged.get_stats(),
    }, args.output)
    print(f"Saved merged model to: {args.output}")
