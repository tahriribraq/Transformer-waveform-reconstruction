"""
Loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class PoissonNLLLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss.
    
    For count data where we predict λ (rate parameter).
    Loss = λ - y * log(λ) + log(y!)
    
    Since log(y!) is constant w.r.t. model params, we use:
    Loss = λ - y * log(λ)
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        log_lambda: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            log_lambda: [batch, seq_len] predicted log(λ)
            target: [batch, seq_len] observed counts
            mask: [batch, seq_len] True for positions to INCLUDE in loss
            
        Returns:
            Scalar loss
        """
        # Clamp for numerical stability
        log_lambda = torch.clamp(log_lambda, min=-20, max=20)
        lambda_pred = torch.exp(log_lambda)
        
        # Poisson NLL: λ - y * log(λ)
        loss = lambda_pred - target * log_lambda
        
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp(min=1)
        else:
            return loss.mean()
        
class NegativeBinomialNLLLoss(nn.Module):
    """
    Negative Binomial Negative Log-Likelihood Loss.
    
    NB distribution is parameterized by:
    - μ (mu): mean
    - r: dispersion parameter (higher r → more Poisson-like)
    
    Variance = μ + μ²/r
    
    This handles overdispersion (variance > mean) common in sparse count data.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        mu: torch.Tensor,
        r: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mu: [batch, seq_len] predicted mean (positive)
            r: [batch, seq_len] dispersion parameter (positive)
            target: [batch, seq_len] observed counts
            mask: [batch, seq_len] True for positions to INCLUDE in loss
            
        Returns:
            Scalar loss
        """
        # Clamp for numerical stability
        mu = mu.clamp(min=self.eps)
        r = r.clamp(min=self.eps, max=1e6)
        
        # NB NLL: -log P(y|μ,r) = -log(Γ(y+r)/(Γ(r)Γ(y+1))) - r*log(r/(r+μ)) - y*log(μ/(r+μ))
        
        # Using torch.lgamma for numerical stability
        loss = (
            torch.lgamma(r) + torch.lgamma(target + 1) - torch.lgamma(target + r)
            + r * torch.log(r + mu) - r * torch.log(r)
            + target * torch.log(r + mu) - target * torch.log(mu + self.eps)
        )
        
        # Apply mask and average
        masked_loss = loss * mask

        return masked_loss.sum() / mask.sum().clamp(min=1)
            
        
class ShapeSimilarityLoss(nn.Module):
    """
    Loss to encourage shape similarity between predicted and target waveforms.
    
    Uses cosine similarity on normalized waveforms (treats them as vectors).
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len] predicted counts
            target: [batch, seq_len] target counts
            weight: [batch, seq_len] position weights
            
        Returns:
            Scalar loss (1 - cosine_similarity)
        """
       # Apply weights
        pred_weighted = pred * weight
        target_weighted = target * weight
        
        # Normalize to unit vectors
        pred_norm = pred_weighted / (pred_weighted.norm(dim=-1, keepdim=True) + self.eps)
        target_norm = target_weighted / (target_weighted.norm(dim=-1, keepdim=True) + self.eps)
        
        # Cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        
        # Loss = 1 - similarity (want to maximize similarity)
        loss = 1 - cos_sim
        
        return loss.mean()

class EarthMoverDistanceLoss(nn.Module):
    """
    1D Earth Mover's Distance (Wasserstein-1) Loss.
    
    Treats waveforms as probability distributions over height and measures
    the "work" needed to transform one into the other.
    
    This penalizes vertical misalignment of peaks proportionally to distance.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len] predicted counts
            target: [batch, seq_len] target counts
            mask: [batch, seq_len] True for positions to INCLUDE
            
        Returns:
            Scalar loss (mean EMD across batch)
        """
        if mask is not None:
            pred = pred * mask.float()
            target = target * mask.float()
        
        # Normalize to probability distributions
        pred_sum = pred.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        target_sum = target.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        
        pred_prob = pred / pred_sum
        target_prob = target / target_sum
        
        # Compute CDFs
        pred_cdf = torch.cumsum(pred_prob, dim=-1)
        target_cdf = torch.cumsum(target_prob, dim=-1)
        
        # EMD = integral of |CDF_pred - CDF_target|
        emd = torch.abs(pred_cdf - target_cdf).mean(dim=-1)
        
        return emd.mean()

class PeakAlignmentLoss(nn.Module):
    """
    Loss that specifically penalizes misalignment of peaks.
    
    Identifies peaks in both predicted and target waveforms and
    penalizes differences in peak locations and heights.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def _find_peaks(
        self,
        waveform: torch.Tensor,
        threshold_ratio: float = 0.1,
        window_half_width: int = 3  # k=1 is the original 3-point behavior
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find peaks in waveform using simple local maximum detection.
        
        Returns:
            peak_mask: [batch, seq_len] True at peak positions
            peak_heights: [batch, seq_len] height at peaks, 0 elsewhere
        """
        batch_size, seq_len = waveform.shape
        
        # Threshold: only consider points above threshold_ratio * max
        max_vals = waveform.max(dim=-1, keepdim=True).values
        threshold = threshold_ratio * max_vals
        above_threshold = waveform > threshold
        
        # Pad by k on each side
        k = window_half_width
        padded = F.pad(waveform, (k, k), value=0)  # [batch, seq_len + 2k]

        # Stack all neighbor columns into [batch, seq_len, 2k] then take max
        neighbors = torch.stack( 
            [padded[:, i : i + seq_len] for i in range(2 * k + 1) if i != k], 
            dim=-1)  # [batch, seq_len, 2k]

        neighbor_max = neighbors.max(dim=-1).values  # [batch, seq_len]

        local_max = waveform > neighbor_max
        
        peak_mask = above_threshold & local_max
        peak_heights = waveform * peak_mask.float()
        
        return peak_mask, peak_heights
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len] predicted counts
            target: [batch, seq_len] target counts
            mask: [batch, seq_len] True for positions to INCLUDE
            
        Returns:
            Scalar loss
        """
        if mask is not None:
            pred = pred * mask.float()
            target = target * mask.float()
        
        # Find peaks in both
        pred_peak_mask, pred_peak_heights = self._find_peaks(pred)
        target_peak_mask, target_peak_heights = self._find_peaks(target)
        
        # Combine peak masks (positions where either has a peak)
        combined_mask = pred_peak_mask | target_peak_mask
        
        # Loss: MSE between predicted and target at peak positions
        if combined_mask.sum() > 0:
            pred_at_peaks = pred * combined_mask.float()
            target_at_peaks = target * combined_mask.float()
            
            # Normalize by target max for scale invariance
            target_max = target.max(dim=-1, keepdim=True).values.clamp(min=self.eps)
            
            pred_normalized = pred_at_peaks / target_max
            target_normalized = target_at_peaks / target_max
            
            loss = F.mse_loss(pred_normalized, target_normalized)
        else:
            loss = torch.tensor(0.0, device=pred.device)
        
        return loss
        
class FHDLoss(nn.Module):
    """
    Differentiable FHD (Foliage Height Diversity / Shannon Index) Loss.
    
    Matches the evaluation function get_FHD() but operates on the fixed
    output grid used by the model.
    
    Key differences from evaluation:
    - Works on fixed grid (no need to sort or digitize)
    - Handles batched data
    - Uses soft operations for differentiability
    """
    
    def __init__(
        self,
        output_heights: np.ndarray,  # The fixed output height grid
        bin_width: float = 1.0,       # Same as evaluation
        eps: float = 1e-8
    ):
        """
        Args:
            output_heights: Array of output heights from model (e.g., 0 to 80m at 0.15m resolution)
            bin_width: Width of height bins in meters (default 1.0m to match evaluation)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.bin_width = bin_width
        self.eps = eps
        
        # Pre-compute bin assignments for the fixed output grid
        # This maps each position in the output grid to a height bin
        output_heights = np.asarray(output_heights)
        
        h_min = np.floor(output_heights.min())
        h_max = np.ceil(output_heights.max())
        bin_edges = np.arange(h_min, h_max + bin_width, bin_width)
        
        # Assign each output position to a bin
        bin_indices = np.digitize(output_heights, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        
        self.num_bins = len(bin_edges) - 1
        self.register_buffer('bin_indices', torch.from_numpy(bin_indices).long())
        
        # Create aggregation matrix: [num_output_positions, num_bins]
        # Each row has a 1 in the column corresponding to its bin
        agg_matrix = np.zeros((len(output_heights), self.num_bins), dtype=np.float32)
        for i, bin_idx in enumerate(bin_indices):
            agg_matrix[i, bin_idx] = 1.0
        
        # Reverse bin order to match your evaluation (binned_counts[::-1])
        agg_matrix = agg_matrix[:, ::-1].copy()
        
        self.register_buffer('agg_matrix', torch.from_numpy(agg_matrix))
    
    def compute_fhd(
        self,
        waveform: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute FHD for a batch of waveforms.
        
        Args:
            waveform: [batch, seq_len] counts at each height position
            mask: [batch, seq_len] valid data mask
            
        Returns:
            fhd: [batch] FHD value for each sample
        """
        # Apply mask
        wf_masked = waveform * mask  # [batch, seq_len]
        
        # Get aggregation matrix on correct device. This handles the case where buffer hasn't been moved yet
        if self.agg_matrix.device != waveform.device:
            agg_matrix = self.agg_matrix.to(waveform.device)
        else:
            agg_matrix = self.agg_matrix
        
        # Aggregate into height bins using matrix multiplication
        # [batch, seq_len] @ [seq_len, num_bins] -> [batch, num_bins]
        binned_counts = torch.matmul(wf_masked, self.agg_matrix)
        
        # Compute proportions (handle zero-sum waveforms)
        total_counts = binned_counts.sum(dim=-1, keepdim=True)  # [batch, 1]
        
        # Avoid division by zero
        total_counts = total_counts.clamp(min=self.eps)
        proportions = binned_counts / total_counts  # [batch, num_bins]
        
        # Shannon index: -sum(p * log(p)) for p > 0
        # Use smooth approximation: p * log(p + eps) for all p
        # This is differentiable and handles zeros gracefully
        log_p = torch.log(proportions + self.eps)
        
        # Only count bins with non-negligible proportions (matching nz = binned_counts[binned_counts > 0])
        # Use soft thresholding for differentiability
        soft_mask = torch.sigmoid((proportions - self.eps) * 1000)  # ~1 if p > eps, ~0 otherwise
        
        fhd = -torch.sum(proportions * log_p * soft_mask, dim=-1)  # [batch]
        
        return fhd
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute FHD loss: MSE between predicted and target FHD values.
        
        Args:
            pred: [batch, seq_len] predicted counts
            target: [batch, seq_len] target counts
            mask: [batch, seq_len] valid data mask
            
        Returns:
            loss: Scalar MSE loss between predicted and target FHD
        """
        pred_fhd = self.compute_fhd(pred, mask)
        target_fhd = self.compute_fhd(target, mask)
        
        return F.mse_loss(pred_fhd, target_fhd)

class FHDCorrelationLoss(nn.Module):
    """
    Alternative: Optimize correlation of FHD values rather than MSE.
    
    This directly targets your goal of FHD correlation > 0.9.
    """
    
    def __init__(
        self,
        output_heights: np.ndarray,
        bin_width: float = 1.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.fhd_computer = FHDLoss(output_heights, bin_width, eps)
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Loss = 1 - correlation(pred_fhd, target_fhd)
        
        Minimizing this maximizes correlation.
        """
        pred_fhd = self.fhd_computer.compute_fhd(pred, mask)
        target_fhd = self.fhd_computer.compute_fhd(target, mask)
        
        # Compute Pearson correlation
        pred_mean = pred_fhd.mean()
        target_mean = target_fhd.mean()
        
        pred_centered = pred_fhd - pred_mean
        target_centered = target_fhd - target_mean
        
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt(
            (pred_centered ** 2).sum() * (target_centered ** 2).sum()
        ).clamp(min=self.eps)
        
        correlation = numerator / denominator
        
        # Loss = 1 - correlation (so minimizing loss maximizes correlation)
        return 1 - correlation

class VCRLoss(nn.Module):
    """
    Differentiable VCR (Vertical Canopy Rugosity / StdBin) Loss.
    
    VCR is the variance of heights weighted by normalized counts:
        H_x = sum(heights * normalized_counts)  # weighted mean height
        VCR = sum(normalized_counts * (heights - H_x)^2)  # weighted variance
    
    """
    
    def __init__(
        self,
        output_heights: np.ndarray,  # The fixed output height grid
        eps: float = 1e-8
    ):
        """
        Args:
            output_heights: Array of output heights from model (e.g., 1 to 80m at 0.15m resolution)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        
        # Register heights as buffer (will move with model.to(device))
        heights_tensor = torch.from_numpy(output_heights.astype(np.float32))
        self.register_buffer('heights', heights_tensor)
    
    def compute_vcr(
        self,
        waveform: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VCR for a batch of waveforms.
        
        Args:
            waveform: [batch, seq_len] counts at each height position
            mask: [batch, seq_len] valid data mask
            
        Returns:
            vcr: [batch] VCR value for each sample
        """
        # Apply mask
        wf_masked = waveform * mask  # [batch, seq_len]
        
        # Ensure heights are on correct device
        if self.heights.device != waveform.device:
            heights = self.heights.to(waveform.device)
        else:
            heights = self.heights
        
        # Normalize counts to get proportions
        total_counts = wf_masked.sum(dim=-1, keepdim=True)  # [batch, 1]
        total_counts = total_counts.clamp(min=self.eps)
        normalized_counts = wf_masked / total_counts  # [batch, seq_len]
        
        # Compute weighted mean height: H_x = sum(heights * normalized_counts)
        # heights: [seq_len], normalized_counts: [batch, seq_len]
        H_x = (heights.unsqueeze(0) * normalized_counts).sum(dim=-1, keepdim=True)  # [batch, 1]
        
        # Compute weighted variance: VCR = sum(normalized_counts * (heights - H_x)^2)
        height_deviation = heights.unsqueeze(0) - H_x  # [batch, seq_len]
        vcr = (normalized_counts * (height_deviation ** 2)).sum(dim=-1)  # [batch]
        
        return vcr
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VCR loss: MSE between predicted and target VCR values.
        
        Args:
            pred: [batch, seq_len] predicted counts
            target: [batch, seq_len] target counts
            mask: [batch, seq_len] valid data mask
            
        Returns:
            loss: Scalar MSE loss between predicted and target VCR
        """
        pred_vcr = self.compute_vcr(pred, mask)
        target_vcr = self.compute_vcr(target, mask)
        
        return F.mse_loss(pred_vcr, target_vcr)

class VCRCorrelationLoss(nn.Module):
    """
    Alternative: Optimize correlation of VCR values rather than MSE.
    
    This directly targets maximizing VCR correlation.
    """
    
    def __init__(
        self,
        output_heights: np.ndarray,
        eps: float = 1e-8
    ):
        super().__init__()
        self.vcr_computer = VCRLoss(output_heights, eps)
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Loss = 1 - correlation(pred_vcr, target_vcr)
        
        Minimizing this maximizes correlation.
        """
        pred_vcr = self.vcr_computer.compute_vcr(pred, mask)
        target_vcr = self.vcr_computer.compute_vcr(target, mask)
        
        # Compute Pearson correlation
        pred_mean = pred_vcr.mean()
        target_mean = target_vcr.mean()
        
        pred_centered = pred_vcr - pred_mean
        target_centered = target_vcr - target_mean
        
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt(
            (pred_centered ** 2).sum() * (target_centered ** 2).sum()
        ).clamp(min=self.eps)
        
        correlation = numerator / denominator
        
        # Loss = 1 - correlation
        return 1 - correlation

class CombinedLoss(nn.Module):
    """
    Keeps zero-region and data-region losses completely separate
    with independent weighting.
    
    This gives you direct control over the trade-off.
    """
    
    def __init__(
        self,
        output_heights: np.ndarray,  # Required for FHD computation
        lambda_data_count: float = 1.0,    # Weight for count loss in data region
        lambda_shape: float = 1.0,      # Weight for shape loss in data region
        lambda_emd: float = 0, # Weight for Earth Mover's Distance loss (vertical alignment)
        lambda_peak: float = 0, # Weight for peak alignment loss (matching peak positions)
        lambda_zero_penalty: float = 0.3,   # Weight for zero-region penalty
        lambda_fhd: float = 0.6,   # Weight for FHD loss
        lambda_vcr: float = 0.5,   # Weight for VCR loss
        bin_width: float = 1.0,	   # Vertical binning interval for FHD and VCR	
        use_correlation: bool = False,  # If True, optimize correlation instead of MSE
        eps: float = 1e-8
    ):
        super().__init__()
        
        self.lambda_data_count = lambda_data_count
        self.lambda_shape = lambda_shape
        self.lambda_zero_penalty = lambda_zero_penalty
        self.lambda_emd = lambda_emd
        self.lambda_peak = lambda_peak
        self.lambda_fhd = lambda_fhd   
        self.lambda_vcr = lambda_vcr
        self.eps = eps
        
        self.nb_loss = NegativeBinomialNLLLoss(eps=eps)
        self.shape_loss = ShapeSimilarityLoss(eps=eps)
        self.emd_loss = EarthMoverDistanceLoss(eps=eps)
        self.peak_loss = PeakAlignmentLoss(eps=eps)
        
        # FHD and VCR losses
        if use_correlation:
            self.fhd_loss = FHDCorrelationLoss(output_heights, bin_width, eps)
            #self.vcr_loss = VCRCorrelationLoss(output_heights, bin_width, eps) # binned version
            self.vcr_loss = VCRCorrelationLoss(output_heights, eps)
        else:
            self.fhd_loss = FHDLoss(output_heights, bin_width, eps)
            #self.vcr_loss = VCRLoss(output_heights, bin_width, eps) # binned version
            self.vcr_loss = VCRLoss(output_heights, eps)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        target: torch.Tensor,
        data_mask: torch.Tensor  # 1.0 where target > 0 OR within valid height range
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dict with 'mu', 'r' from model
            target: [batch, seq_len] target counts
            data_mask: [batch, seq_len] 1.0 for data region, 0.0 for zero region
        """
        mu = predictions['mu']
        r = predictions['r']
        
        losses = {}
        
        # === Data Region Loss ===
        # Only compute count loss where we have actual data
        data_region = data_mask
        if data_region.sum() > 0:
            losses['data_count'] = self.nb_loss(mu, r, target, data_region)
        else:
            losses['data_count'] = torch.tensor(0.0, device=mu.device)
        
        # === Shape (on full waveform) ===
        # Use data_mask to focus on data region but don't completely ignore zeros
        full_weight = data_mask + 0.1 * (1 - data_mask)  # Small weight for zeros
        losses['shape'] = self.shape_loss(mu, target, full_weight)

        # === Zero Region Loss ===
        # Penalize any non-zero predictions where target is zero
        zero_region = 1.0 - data_mask
        
        # Simple L1 penalty on predictions in zero region
        # (target is 0 there, so this penalizes |mu - 0| = |mu|)
        zero_penalty = (mu * zero_region).abs()
        if zero_region.sum() > 0:
            losses['zero_penalty'] = zero_penalty.sum() / zero_region.sum()
        else:
            losses['zero_penalty'] = torch.tensor(0.0, device=mu.device)
            
        # === FHD Loss ===
        if self.lambda_fhd > 0:
            losses['fhd'] = self.fhd_loss(mu, target, data_mask)
        else:
            losses['fhd'] = torch.tensor(0.0, device=mu.device)
        
        # === VCR Loss ===
        if self.lambda_vcr > 0:
            losses['vcr'] = self.vcr_loss(mu, target, data_mask)
        else:
            losses['vcr'] = torch.tensor(0.0, device=mu.device)
            
        # === EMD loss (optional) === #
        if self.lambda_emd > 0:
            losses['emd'] = self.emd_loss(mu, target, data_mask)
        else:
            losses['emd'] = torch.tensor(0.0, device=mu.device)

        # === Peak alignment loss (optional) === #
        if self.lambda_peak > 0:
            losses['peak'] = self.peak_loss(mu, target, data_mask)
        else:
            losses['peak'] = torch.tensor(0.0, device=mu.device)
        
        # === Combined ===
        losses['total'] = (
            self.lambda_data_count * losses['data_count'] +
            self.lambda_shape * losses['shape'] +
            self.lambda_zero_penalty * losses['zero_penalty'] +
            self.lambda_fhd * losses['fhd'] +
            self.lambda_vcr * losses['vcr'] +
            self.lambda_emd * losses['emd'] +
            self.lambda_peak * losses['peak']
        )
        
        return losses
