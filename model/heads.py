"""
Output prediction heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CountPredictionHead(nn.Module):
    """
    Output head for predicting ALS return counts.
    
    Supports two modes:
    1. Poisson: Predicts log(λ) for Poisson distribution
    2. Negative Binomial: Predicts μ and r for NB distribution
    
    The NB distribution is more flexible for overdispersed count data
    (variance > mean), which is common in sparse, peaky waveforms.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        distribution: str = "negative_binomial"
    ):
        """
        Args:
            embed_dim: Input dimension from decoder
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            distribution: "poisson" or "negative_binomial"
        """
        super().__init__()
        
        self.distribution = distribution
        
        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        if distribution == "poisson":
            # Output: log(λ) for Poisson
            self.output_layer = nn.Linear(hidden_dim, 1)
        elif distribution == "negative_binomial":
            # Output: log(μ) and log(r) for Negative Binomial
            self.mu_layer = nn.Linear(hidden_dim, 1)
            self.r_layer = nn.Linear(hidden_dim, 1)
            # Initialize r to give reasonable initial variance
            nn.init.constant_(self.r_layer.bias, 2.0)  # r ≈ exp(2) ≈ 7.4
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def forward(
        self,
        x: torch.Tensor
    ) -> dict:
        """
        Args:
            x: [batch, num_positions, embed_dim] decoder output
            
        Returns:
            dict with distribution parameters:
                - For Poisson: {'log_lambda': [batch, num_positions]}
                - For NB: {'log_mu': ..., 'log_r': ..., 'mu': ..., 'r': ...}
        """
        features = self.feature_net(x)
        
        if self.distribution == "poisson":
            log_lambda = self.output_layer(features).squeeze(-1)
            return {
                'log_lambda': log_lambda,
                'lambda': torch.exp(log_lambda)
            }
        else:  # negative_binomial
            log_mu = self.mu_layer(features).squeeze(-1)
            log_r = self.r_layer(features).squeeze(-1)
            
            mu = F.softplus(log_mu)  # Ensure positive
            r = F.softplus(log_r)    # Ensure positive
            
            return {
                'log_mu': log_mu,
                'log_r': log_r,
                'mu': mu,
                'r': r
            }
    
    def get_mean_prediction(self, output: dict) -> torch.Tensor:
        """Get the mean predicted count (for both distributions)."""
        if self.distribution == "poisson":
            return output['lambda']
        else:
            return output['mu']