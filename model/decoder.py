"""
Transformer decoder for ALS waveform prediction.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import MultiHeadAttention
from .encoder import FeedForward

class DecoderLayer(nn.Module):
    """
    Single transformer decoder layer.
    
    Architecture:
        x → LayerNorm → Self-Attention → Dropout → Add →
          → LayerNorm → Cross-Attention → Dropout → Add →
          → LayerNorm → FFN → Dropout → Add → output
    
    The cross-attention attends to the encoder output (LVIS representation).
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
        
        # Self-attention over output queries
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-attention to encoder output
        self.cross_attn = MultiHeadAttention(
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
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, num_queries, embed_dim] output queries
            encoder_output: [batch, src_len, embed_dim] LVIS encoder output
            encoder_padding_mask: [batch, src_len] True for padded LVIS positions
            
        Returns:
            output: [batch, num_queries, embed_dim]
            self_attn_weights: [batch, num_heads, num_queries, num_queries]
            cross_attn_weights: [batch, num_heads, num_queries, src_len]
        """
        # Self-attention over output positions
        residual = x
        x = self.norm1(x)
        x, self_attn_weights = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x
        
        # Cross-attention to encoder output
        residual = x
        x = self.norm2(x)
        x, cross_attn_weights = self.cross_attn(
            query=x,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=encoder_padding_mask
        )
        x = self.dropout(x)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, self_attn_weights, cross_attn_weights
    
class ALSDecoder(nn.Module):
    """
    Transformer decoder for predicting ALS waveforms.

    Takes output queries (anchored to absolute heights) and cross-attends
    to the encoded LVIS representation to predict ALS return counts
    at each height bin.
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
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
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
        queries: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            queries: [batch, num_queries, embed_dim] height-anchored output queries
            encoder_output: [batch, src_len, embed_dim] encoded LVIS waveform
            encoder_padding_mask: [batch, src_len] True for padded LVIS positions
            return_attention: whether to return attention weights
            
        Returns:
            output: [batch, num_queries, embed_dim] decoded representations
            attention_weights: dict with 'self_attn' and 'cross_attn' lists if requested
        """
        attention_weights = {'self_attn': [], 'cross_attn': []} if return_attention else None
        
        x = queries
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(
                x=x,
                encoder_output=encoder_output,
                encoder_padding_mask=encoder_padding_mask
            )
            
            if return_attention:
                attention_weights['self_attn'].append(self_attn)
                attention_weights['cross_attn'].append(cross_attn)
        
        x = self.final_norm(x)
        
        return x, attention_weights