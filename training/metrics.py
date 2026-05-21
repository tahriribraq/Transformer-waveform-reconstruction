"""
Evaluation metrics.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

class Metrics:
    """Compute evaluation metrics for waveform prediction."""
    
    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Root Mean Squared Error."""
        if mask is not None:
            pred = pred[mask > 0]
            target = target[mask > 0]
        mse = F.mse_loss(pred, target)
        return torch.sqrt(mse).item()
    
    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Mean Absolute Error."""
        if mask is not None:
            pred = pred[mask > 0]
            target = target[mask > 0]
        return F.l1_loss(pred, target).item()
    
    @staticmethod
    def correlation(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Pearson correlation coefficient."""
        if mask is not None:
            pred = pred[mask > 0]
            target = target[mask > 0]
        
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
        
        if denominator < 1e-8:
            return 0.0
        
        return (numerator / denominator).item()
    
    @staticmethod
    def n_correlation_rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Correlation after max-normalizing each waveform and concatenating."""
        if mask is not None:
            valid = mask > 0
        else:
            valid = torch.ones_like(pred, dtype=torch.bool)
        
        norm_pred_all = []
        norm_target_all = []
        
        for i in range(pred.shape[0]):
            sample_valid = valid[i]
            if sample_valid.sum() == 0:
                continue
            
            p = pred[i][sample_valid]
            t = target[i][sample_valid]
            
            p_max = p.max()
            t_max = t.max()
            if p_max <= 1e-8 or t_max <= 1e-8:
                continue
            
            norm_pred_all.append(p / p_max)
            norm_target_all.append(t / t_max)
        
        if len(norm_pred_all) == 0:
            return 0.0
        
        pred_flat = torch.cat(norm_pred_all, dim=0)
        target_flat = torch.cat(norm_target_all, dim=0)
        
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
        
        if denominator < 1e-8:
            return 0.0, 0.0
        
        n_mse = F.mse_loss(pred_flat, target_flat)
        n_rmse = torch.sqrt(n_mse).item()
        
        return (numerator / denominator).item(), n_rmse
    
    @staticmethod
    def r_squared(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Coefficient of determination (R²)."""
        if mask is not None:
            pred = pred[mask > 0]
            target = target[mask > 0]
        
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        
        if ss_tot < 1e-8:
            return 0.0
        
        return (1 - ss_res / ss_tot).item()
    
    @staticmethod
    def zero_region_accuracy(
        pred: torch.Tensor,
        target: torch.Tensor,
        data_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        Accuracy of predicting zeros in zero regions.
        
        Measures what fraction of zero-region predictions are below threshold.
        """
        zero_mask = 1.0 - data_mask
        if zero_mask.sum() == 0:
            return 1.0
        
        zero_region_preds = pred[zero_mask > 0]
        correct = (zero_region_preds < threshold).float().mean()
        
        return correct.item()
    
    @staticmethod
    def compute_all(
        pred: torch.Tensor,
        target: torch.Tensor,
        data_mask: torch.Tensor,
        heights: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all metrics at once."""
        
        n_corr, n_rmse = Metrics.n_correlation_rmse(pred, target, data_mask)
        
        return {
            'rmse': Metrics.rmse(pred, target, data_mask),
            'mae': Metrics.mae(pred, target, data_mask),
            'correlation': Metrics.correlation(pred, target, data_mask),
            'n_correlation': n_corr,
            'n_rmse': n_rmse,
            'r_squared': Metrics.r_squared(pred, target, data_mask),
            'zero_accuracy': Metrics.zero_region_accuracy(pred, target, data_mask)
        }
