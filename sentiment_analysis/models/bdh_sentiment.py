"""
BDH model adapted for sentiment classification.
Self-contained implementation based on the BDH architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BDHSentimentConfig:
    """Configuration for BDH sentiment model."""
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 32
    vocab_size: int = 30522
    dropout: float = 0.2
    num_classes: int = 5
    max_seq_length: int = 128
    pooling_strategy: str = 'mean'  # 'mean', 'max', or 'cls'


def get_freqs(n, theta, dtype):
    """Generate frequency values for RoPE."""
    def quantize(t, q=2):
        return (t / q).floor() * q
    
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    """BDH Attention mechanism with RoPE."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        # Register buffer for frequencies
        freqs = get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        self.register_buffer('freqs', freqs)
    
    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin
    
    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)
    
    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()
        
        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR
        
        # Current attention
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDHCore(nn.Module):
    """Core BDH model for sequence encoding."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        
        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        """
        Forward pass through BDH layers.
        
        Args:
            idx: (batch_size, seq_len) token indices
            
        Returns:
            hidden_states: (batch_size, seq_len, n_embd)
        """
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        x = self.embed(idx).unsqueeze(1)  # B, 1, T, D
        x = self.ln(x)
        
        # Run through BDH layers
        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N
            
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N
            
            xy_sparse = self.drop(xy_sparse)
            
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)
        
        # x shape: (B, 1, T, D) -> squeeze to (B, T, D)
        hidden_states = x.squeeze(1)
        
        return hidden_states


class BDHSentiment(nn.Module):
    """
    BDH model adapted for sentiment classification.
    
    Architecture:
    1. Token embeddings via BDH core model
    2. Sequence encoding through BDH layers
    3. Pooling layer (mean/max/cls)
    4. Classification head
    """
    
    def __init__(self, config: BDHSentimentConfig):
        super().__init__()
        self.config = config
        
        # BDH core for sequence encoding
        self.bdh_core = BDHCore(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd, config.num_classes)
        )
        
        self.pooling_strategy = config.pooling_strategy
    
    def pool_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence representations.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            pooled: (batch_size, hidden_dim)
        """
        if self.pooling_strategy == 'cls':
            # Use first token (CLS-like)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == 'max':
            # Max pooling
            if attention_mask is not None:
                # Mask padded positions
                hidden_states = hidden_states.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, -1e9
                )
            return torch.max(hidden_states, dim=1)[0]
        
        else:  # mean pooling (default)
            if attention_mask is not None:
                # Masked mean pooling
                sum_hidden = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
                sum_mask = attention_mask.sum(dim=1, keepdim=True)
                return sum_hidden / sum_mask.clamp(min=1e-9)
            else:
                return hidden_states.mean(dim=1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) - optional for training
            
        Returns:
            logits: (batch_size, num_classes)
            loss: scalar (if labels provided)
        """
        # Get BDH embeddings
        hidden_states = self.bdh_core(input_ids)
        
        # Pool sequence
        pooled = self.pool_sequence(hidden_states, attention_mask)
        
        # Classification
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return logits, loss
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
