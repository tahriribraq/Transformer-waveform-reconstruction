"""
Learned queries for decoder.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .embeddings import LearnedHeightEmbedding, FourierHeightEncoding

class ALSOutputQueries(nn.Module):
    """
    Learned queries for the decoder, anchored to absolute heights.

    These serve as the "questions" the decoder asks of the encoded LVIS waveform:
    "What should the ALS return count be at height h?"
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        height_range: Tuple[float, float] = (1, 100.0),
        height_resolution: float = 0.15, # meters per bin
        num_height_frequencies: int = 32,
        dropout: float = 0.1,
        fourier_encoder: Optional[FourierHeightEncoding] = None,
        fusion_type: str = 'MLP'  # 'Add' or 'MLP'
    ):
        super().__init__()
        
        self.height_min = height_range[0]
        self.height_max = height_range[1]
        self.height_resolution = height_resolution
        self.embed_dim = embed_dim
        # Calculate number of output positions
        self.num_positions = int((height_range[1] - height_range[0]) / height_resolution) + 1
        # Height embedding (same architecture as encoder for consistency)
        self.height_embedding = LearnedHeightEmbedding(
            embed_dim=embed_dim,
            num_frequencies=num_height_frequencies,
            height_range=height_range,
            dropout=dropout,
            fourier_encoder=fourier_encoder
        )
        
        self.fusion_type = fusion_type

        # Learnable content queries (what to predict at each position). These learn position-agnostic patterns about ALS returns
        self.content_queries = nn.Parameter(
            torch.randn(1, self.num_positions, embed_dim) * 0.02
        )

        # Final layer norm after summation
        self.final_norm = nn.LayerNorm(embed_dim)

        # Combine height embedding with content queries
        self.query_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Pre-compute height values for the fixed grid
        heights = torch.linspace(height_range[0], height_range[1], self.num_positions)
        self.register_buffer('output_heights', heights)
        
    def forward(
        self,
        batch_size: int,
        height_subset: Optional[Tuple[float, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate output queries for the decoder.
        
        Args:
            batch_size: number of samples in batch
            height_subset: optional (min, max) to select a height range subset
            
        Returns:
            queries: [batch, num_positions, embed_dim]
            heights: [num_positions] height values for each query
            valid_mask: [num_positions] boolean mask for valid positions
        """
        if height_subset is not None:
            # Select subset of positions within height range
            valid_mask = (
                (self.output_heights >= height_subset[0]) &
                (self.output_heights <= height_subset[1])
            )
            indices = torch.where(valid_mask)[0]
            heights = self.output_heights[indices]
            content = self.content_queries[:, indices, :]
        else:
            heights = self.output_heights
            content = self.content_queries
            valid_mask = torch.ones(self.num_positions, dtype=torch.bool, 
                                    device=heights.device)
        # Get height embeddings
        height_embed = self.height_embedding(heights.unsqueeze(0)) # [1, num_pos, embed_dim]

        if self.fusion_type == 'Add':
            # Simple addition fusion
            queries = content.expand(batch_size, -1, -1) + height_embed.expand(batch_size, -1, -1)
            queries = self.final_norm(queries)
        
        if self.fusion_type == 'MLP':
            # Fuse content queries with height embeddings
            combined = torch.cat([
                content.expand(batch_size, -1, -1),
                height_embed.expand(batch_size, -1, -1)
            ], dim=-1)
        
            queries = self.query_fusion(combined)
        
        return queries, heights, valid_mask