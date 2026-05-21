"""
Main LVIS to ALS transformer model.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .embeddings import LVISInputEmbedding, FourierHeightEncoding
from .encoder import LVISEncoder
from .decoder import ALSDecoder
from .queries import ALSOutputQueries
from .heads import CountPredictionHead

class LVIS2ALSTransformer(nn.Module):
    """
    Complete transformer model for LVIS to ALS waveform translation.
    
    Architecture:
        LVIS waveform → Input Embedding → Encoder → 
        Output Queries + Encoder Output → Decoder → 
        Count Prediction Head → ALS count predictions
    """
    
    def __init__(
        self,
        # Embedding parameters
        embed_dim: int = 256,
        lvis_height_range: Tuple[float, float] = (-15.0, 85.0),
        als_height_range: Tuple[float, float] = (1.0, 100.0),
        height_resolution: float = 0.15,
        global_max_count: float = 4096.0,
        global_max_sum: float = 80684.0,
        num_height_frequencies: int = 32,
        use_auxiliary_features: bool = True,
        use_cls_token: bool = True,
        als_query_fusion: str = 'Add',  # 'Add' or 'MLP'
        # Encoder parameters
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        encoder_ffn_dim: int = 1024,
        # Decoder parameters
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ffn_dim: int = 1024,
        # Output parameters
        output_distribution: str = "negative_binomial",
        # Regularization
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.lvis_height_range = lvis_height_range
        self.als_height_range = als_height_range
        self.height_resolution = height_resolution
        self.use_cls_token = use_cls_token
        
        # Shared Fourier encoder for both LVIS and ALS. This ensures the same absolute height gets the same base encoding
        self.shared_fourier_encoder = FourierHeightEncoding(
            num_frequencies=num_height_frequencies,
            learnable_frequencies=True
        )

        # Input embedding for LVIS waveforms
        self.input_embedding = LVISInputEmbedding(
            embed_dim=embed_dim,
            height_range=lvis_height_range,
            global_max_count=global_max_count,
            global_max_sum=global_max_sum,
            num_height_frequencies=num_height_frequencies,
            dropout=dropout,
            use_auxiliary_features=use_auxiliary_features,
            fourier_encoder=self.shared_fourier_encoder
        )
        
        
        # Output queries for ALS prediction
        self.output_queries = ALSOutputQueries(
            embed_dim=embed_dim,
            height_range=als_height_range,
            height_resolution=height_resolution,
            num_height_frequencies=num_height_frequencies,
            dropout=dropout,
            fourier_encoder=self.shared_fourier_encoder,
            fusion_type=als_query_fusion
        )
        
        # Encoder
        self.encoder = LVISEncoder(
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            ffn_dim=encoder_ffn_dim,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = ALSDecoder(
            embed_dim=embed_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ffn_dim=decoder_ffn_dim,
            dropout=dropout
        )
        
        # Count prediction head
        self.count_head = CountPredictionHead(
            embed_dim=embed_dim,
            hidden_dim=embed_dim // 2,
            dropout=dropout,
            distribution=output_distribution
        )
        
        # Store number of output positions
        self.num_output_positions = self.output_queries.num_positions
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        photon_counts: torch.Tensor,
        heights: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> dict:
        """
        Forward pass for LVIS to ALS translation.
        
        Args:
            photon_counts: [batch, seq_len] LVIS photon counts
            heights: [batch, seq_len] heights in meters for each LVIS bin
            padding_mask: [batch, seq_len] True for padded positions
            return_attention: whether to return attention weights
            
        Returns:
            dict containing:
                - 'predictions': dict with distribution parameters
                - 'mean_counts': [batch, num_output_positions] mean predicted counts
                - 'output_heights': [num_output_positions] height for each output bin
                - 'encoder_attention': list of attention weights (if requested)
                - 'decoder_attention': dict of attention weights (if requested)
        """
        batch_size = photon_counts.shape[0]
        
        # Embed LVIS input
        embedded, updated_mask = self.input_embedding(
            photon_counts=photon_counts,
            heights=heights,
            padding_mask=padding_mask,
            add_cls_token=self.use_cls_token
        )
        
        # Encode LVIS waveform
        encoder_output, encoder_attn = self.encoder(
            x=embedded,
            padding_mask=updated_mask,
            return_attention=return_attention
        )
        
        # Generate output queries
        queries, output_heights, _ = self.output_queries(batch_size=batch_size)
        
        # Decode to get ALS representations
        decoder_output, decoder_attn = self.decoder(
            queries=queries,
            encoder_output=encoder_output,
            encoder_padding_mask=updated_mask,
            return_attention=return_attention
        )
        
        # Predict counts
        predictions = self.count_head(decoder_output)
        mean_counts = self.count_head.get_mean_prediction(predictions)
        
        result = {
            'predictions': predictions,
            'mean_counts': mean_counts,
            'output_heights': output_heights
        }
        
        if return_attention:
            result['encoder_attention'] = encoder_attn
            result['decoder_attention'] = decoder_attn
        
        return result
    
    def get_output_height_indices(
        self,
        target_heights: torch.Tensor
    ) -> torch.Tensor:
        """
        Map target heights to output position indices. Useful for extracting predictions at specific heights for loss computation.
        
        Args:
            target_heights: [batch, target_len] heights in meters          
        Returns:
            indices: [batch, target_len] indices into output positions
        """
        # Convert height to index
        indices = ((target_heights - self.height_range[0]) / self.height_resolution).long()
        indices = torch.clamp(indices, 0, self.num_output_positions - 1)
        
        return indices