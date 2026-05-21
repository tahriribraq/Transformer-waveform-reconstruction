"""
Embedding layers for LVIS input and height encoding.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

class FourierHeightEncoding(nn.Module):
    """
    Encodes continuous height values using Fourier features.
    
    This creates a rich representation of height that captures both
    fine-grained positional information and coarse spatial context.
    """
    
    def __init__(
        self,
        num_frequencies: int = 32,
        min_wavelength: float = 0.3,    # meters (2x bin resolution)
        max_wavelength: float = 50.0,   # meters (coarse canopy scale)
        learnable_frequencies: bool = False
    ):
        super().__init__()
        
        self.num_frequencies = num_frequencies
        
        # Log-spaced frequencies for multi-scale representation
        # Frequency = 2π / wavelength
        wavelengths = torch.logspace(
            math.log10(min_wavelength),
            math.log10(max_wavelength),
            num_frequencies
        )
        frequencies = 2 * math.pi / wavelengths
        
        if learnable_frequencies:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer('frequencies', frequencies)
    
    @property
    def output_dim(self) -> int:
        # sin and cos for each frequency, plus raw height
        return 2 * self.num_frequencies + 1
    
    def forward(self, heights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heights: [batch, seq_len] or [batch, seq_len, 1] in meters
            
        Returns:
            Fourier features: [batch, seq_len, 2 * num_frequencies + 1]
        """
        if heights.dim() == 3:
            heights = heights.squeeze(-1)
        
        # [batch, seq_len, num_frequencies]
        angles = heights.unsqueeze(-1) * self.frequencies
        
        # Concatenate sin, cos, and normalized raw height
        fourier_features = torch.cat([
            torch.sin(angles),
            torch.cos(angles),
            heights.unsqueeze(-1) / 50.0  # rough normalization
        ], dim=-1)
        
        return fourier_features
    
class LearnedHeightEmbedding(nn.Module):
    """
    Hybrid height embedding: Fourier features → MLP → learned embedding.
    
    This combines the smoothness of continuous encodings with the
    flexibility of learned representations.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_frequencies: int = 32,
        height_range: Tuple[float, float] = (-15.0, 85.0),
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
        fourier_encoder: Optional[FourierHeightEncoding] = None
    ):
        super().__init__()
        
        self.height_min = height_range[0]
        self.height_max = height_range[1]
        self.embed_dim = embed_dim
        
        # Use shared or create new Fourier encoder
        if fourier_encoder is not None:
            self.fourier_encoder = fourier_encoder
        else:
            self.fourier_encoder = FourierHeightEncoding(
                num_frequencies=num_frequencies,
                learnable_frequencies=True
            )
        
        # MLP to project Fourier features to embedding space
        fourier_dim = self.fourier_encoder.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, heights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heights: [batch, seq_len] height values in meters
            
        Returns:
            embeddings: [batch, seq_len, embed_dim]
        """

        heights_clamped = torch.clamp(heights, self.height_min, self.height_max) # Clamp to valid range
        fourier_features = self.fourier_encoder(heights_clamped)  # Fourier encode
        embeddings = self.mlp(fourier_features) # Project through MLP   
        embeddings = self.layer_norm(embeddings) # Normalize
        
        return embeddings
    
class LVISInputEmbedding(nn.Module):
    """
    Complete input representation for LVIS waveforms.
    
    Combines:
    1. Photon count embedding (value information)
    2. Height embedding (positional/physical information)
    3. Optional auxiliary features
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        height_range: Tuple[float, float] = (-15.0, 85.0),
        global_max_count: float = 4094.0,
        global_max_sum: float = 80684.0,
        num_height_frequencies: int = 32,
        dropout: float = 0.1,
        use_auxiliary_features: bool = True,
        fourier_encoder: Optional[FourierHeightEncoding] = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.height_range = height_range
        self.global_max_count = global_max_count
        self.global_max_sum = global_max_sum
        self.use_auxiliary_features = use_auxiliary_features
        
        # Height embedding (positional)
        self.height_embedding = LearnedHeightEmbedding(
            embed_dim=embed_dim,
            num_frequencies=num_height_frequencies,
            height_range=height_range,
            dropout=dropout,
            fourier_encoder=fourier_encoder
        )
        
        # Photon count embedding
        # Input features: shape (per-waveform max normalized), log intensity (absolute), waveform energy (sum-based, broadcast)
        
        count_feature_dim = 3
        if use_auxiliary_features:
            count_feature_dim += 2
        
        # Project count features to embed_dim
        self.count_projection = nn.Sequential(
            nn.Linear(count_feature_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Optional [CLS] token for global context
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)    
        # Final layer norm after summation
        self.final_norm = nn.LayerNorm(embed_dim)
        
    def _compute_count_features(
        self,
        counts: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-scale count features.
        
        Args:
            counts: [batch, seq_len] photon counts
            padding_mask: [batch, seq_len] True for padded positions
            
        Returns:
            features: [batch, seq_len, num_features]
        """
        batch_size, seq_len = counts.shape
        
        # Mask padded values for aggregation computations
        if padding_mask is not None:
            masked_counts = counts.masked_fill(padding_mask, 0)
        else:
            masked_counts = counts
        
        # Per-waveform statistics
        waveform_max = masked_counts.max(dim=-1, keepdim=True).values.clamp(min=1)
        waveform_sum = masked_counts.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # =====================================================================
        # Feature 1: Shape (per-waveform max normalized)
        # =====================================================================
        # Emphasizes the profile shape, independent of overall brightness
        # All waveforms have values in [0, 1] with peak at 1
        counts_shape = counts / waveform_max
        
        # =====================================================================
        # Feature 2: Log intensity (absolute, per-bin)
        # =====================================================================
        # Preserves absolute intensity information with good dynamic range. Useful for distinguishing dim vs. bright bins in absolute terms
        counts_log = torch.log1p(counts) / math.log(self.global_max_count + 1)
        
        # =====================================================================
        # Feature 3: Waveform energy (sum-based, broadcast to all bins)
        # =====================================================================
        # Captures total returned energy for the entire waveform as a measure of overall waveform brightness
        # Using log scale for numerical stability and to handle the range
        # (min=6627, avg=53968, max=80684)
        waveform_energy = torch.log1p(waveform_sum) / math.log(self.global_max_sum + 1)
        waveform_energy = waveform_energy.expand(-1, seq_len)  # [batch, seq_len]
        
        features = [counts_shape, counts_log, waveform_energy]
        
        # =====================================================================
        # Auxiliary features (optional)
        # =====================================================================
        if self.use_auxiliary_features:
            # Feature 4: Local gradient (normalized by waveform max)
            # Helps identify peaks (gradient crosses zero), slopes, plateaus
            gradient = torch.zeros_like(counts)
            gradient[:, 1:] = counts[:, 1:] - counts[:, :-1]
            gradient = gradient / waveform_max
            
            # Feature 5: Distance from peak (normalized by sequence length)
            # Provides relative position context within the waveform
            max_indices = masked_counts.argmax(dim=-1, keepdim=True)
            positions = torch.arange(seq_len, device=counts.device).unsqueeze(0)
            
            if padding_mask is not None:
                seq_lengths = (~padding_mask).sum(dim=-1, keepdim=True).float()
            else:
                seq_lengths = torch.full((batch_size, 1), seq_len, 
                                        dtype=torch.float, device=counts.device)
            
            distance_from_peak = (positions - max_indices).float() / seq_lengths.clamp(min=1)
            
            features.extend([gradient, distance_from_peak])
        
        return torch.stack(features, dim=-1)
    
    def forward(
        self,
        photon_counts: torch.Tensor,
        heights: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        add_cls_token: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create input embeddings for LVIS waveforms.
        
        Args:
            photon_counts: [batch, seq_len] raw photon counts
            heights: [batch, seq_len] height in meters for each bin
            padding_mask: [batch, seq_len] True for padded positions
            add_cls_token: whether to prepend [CLS] token
            
        Returns:
            embeddings: [batch, seq_len (+1 if cls), embed_dim]
            updated_padding_mask: [batch, seq_len (+1 if cls)] or None
        """
        batch_size = photon_counts.shape[0]
        
        # Compute count features and project to embedding space
        count_features = self._compute_count_features(photon_counts, padding_mask)
        count_embed = self.count_projection(count_features)
        
        # Get height embeddings
        height_embed = self.height_embedding(heights)
        
        # Fuse via summation
        embeddings = count_embed + height_embed
        embeddings = self.final_norm(embeddings)
        
        # Zero out padded positions
        if padding_mask is not None:
            embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0)
        
        # Add [CLS] token
        if add_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)
            
            if padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, 
                                       device=padding_mask.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        return embeddings, padding_mask