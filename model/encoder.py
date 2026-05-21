"""
Transformer encoder for LVIS waveforms.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer.
    
    Architecture:
        x → LayerNorm → Self-Attention → Dropout → Add → 
          → LayerNorm → FFN → Dropout → Add → output
    
    Uses pre-norm architecture (LayerNorm before attention/FFN) for
    more stable training with deep networks.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            padding_mask: [batch, seq_len] True for padded positions
            
        Returns:
            output: [batch, seq_len, embed_dim]
            attn_weights: [batch, num_heads, seq_len, seq_len]
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x, attn_weights = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.dropout(x)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, attn_weights
    
class LVISEncoder(nn.Module):
    """
    Transformer encoder for LVIS waveforms.
    
    Takes embedded LVIS waveforms and produces contextualized representations
    where each height bin is aware of the full waveform context.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            x: [batch, seq_len, embed_dim] embedded LVIS waveform
            padding_mask: [batch, seq_len] True for padded positions
            return_attention: whether to return attention weights from all layers
            
        Returns:
            output: [batch, seq_len, embed_dim] contextualized representations
            attention_weights: list of [batch, num_heads, seq_len, seq_len] if requested
        """
        attention_weights = [] if return_attention else None
        
        for layer in self.layers:
            x, attn = layer(x, padding_mask=padding_mask)
            if return_attention:
                attention_weights.append(attn)
        
        x = self.final_norm(x)
        
        return x, attention_weights