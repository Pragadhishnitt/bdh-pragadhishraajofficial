"""Baseline models for comparison."""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, Tuple


class DistilBERTSentiment(nn.Module):
    """
    DistilBERT fine-tuned for sentiment classification.
    Baseline for comparison with BDH.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 5,
        dropout: float = 0.2,
        freeze_base: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Load pre-trained DistilBERT
        self.distilbert = AutoModel.from_pretrained(model_name)
        
        if freeze_base:
            # Freeze base model parameters
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        hidden_size = self.distilbert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
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
            labels: (batch_size,)
            
        Returns:
            logits: (batch_size, num_classes)
            loss: scalar (if labels provided)
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        
        return logits, loss
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
