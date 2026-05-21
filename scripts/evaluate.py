"""
Main evaluation script for LVIS to ALS model.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --batch_size 16 --epochs 50
"""

import argparse
from logging import config
import sys
from pathlib import Path
import pickle
from xml.parsers.expat import model
from xml.parsers.expat import model
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from data.dataset import create_data_splits
from model import LVIS2ALSTransformer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate LVIS to ALS model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--lvis_file', type=str, default='LVIS_wf_list.pkl',
                       help='LVIS waveforms file')
    parser.add_argument('--als_file', type=str, default='AOP_wf_list.pkl',
                       help='ALS waveforms file')
    
    # Load checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    return parser.parse_args()

def load_data(args):
    """Load LVIS and ALS waveforms."""
    data_dir = Path(args.data_dir)
    
    print(f"Loading data from {data_dir}")
    # Load LVIS waveforms from file
    lvis_path = data_dir / args.lvis_file
    with open(lvis_path, 'rb') as f:
        lvis_waveforms = pickle.load(f)

    # Load ALS waveforms from file
    als_path = data_dir / args.als_file
    with open(als_path, 'rb') as f:
        als_waveforms = pickle.load(f)
    
    print(f"Loaded {len(lvis_waveforms)} waveform pairs")
    
    return lvis_waveforms, als_waveforms

def compute_lvis_aop_metrics(LVIS_wf_list, AOP_wf_list, config):
    lvis_counts_all, lvis_counts_norm_all = [], []
    aop_counts_all, aop_counts_norm_all = [], []
    
    for lvis_wf, aop_wf in tqdm(zip(LVIS_wf_list, AOP_wf_list)):
        # Only compare if both waveforms are non-empty
        if lvis_wf.shape[0] == 0 or aop_wf.shape[0] == 0:
            continue
        # Create fixed output height grid
        lvis_range = config.lvis_height_range
        als_range = config.als_height_range
        
        height_min = np.min([lvis_range[0], als_range[0]])
        height_max = np.max([lvis_range[1], als_range[1]])
        bin_res = config.height_resolution

        num_bins = int((height_max - height_min) / bin_res) + 1

        # Bin LVIS counts to grid
        lvis_heights = lvis_wf[:, 0]
        lvis_counts = lvis_wf[:, 1]
        lvis_binned = np.zeros(num_bins, dtype=np.float32)
        inds = ((lvis_heights - height_min) / bin_res).astype(int)
        inds = np.clip(inds, 0, num_bins - 1)
        for idx, c in zip(inds, lvis_counts):
            lvis_binned[idx] += c

        # Bin AOP counts to grid
        aop_heights = aop_wf[:, 0]
        aop_counts = aop_wf[:, 1]
        aop_binned = np.zeros(num_bins, dtype=np.float32)
        inds = ((aop_heights - height_min) / bin_res).astype(int)
        inds = np.clip(inds, 0, num_bins - 1)
        for idx, c in zip(inds, aop_counts):
            aop_binned[idx] += c

        lvis_counts_all.append(lvis_binned)
        aop_counts_all.append(aop_binned)
        
        # Normalize each waveform pair by its own max and store for normalized metrics
        lvis_max, aop_max = np.max(lvis_binned), np.max(aop_binned)
        if lvis_max > 0 and aop_max > 0:
            lvis_counts_norm_all.append(lvis_binned / lvis_max)
            aop_counts_norm_all.append(aop_binned / aop_max)		
        	
    # Concatenate all
    lvis_counts_all = np.concatenate(lvis_counts_all)
    aop_counts_all = np.concatenate(aop_counts_all)
    lvis_counts_norm_all = np.concatenate(lvis_counts_norm_all)
    aop_counts_norm_all = np.concatenate(aop_counts_norm_all)
    
    # Raw metrics
    rmse = np.sqrt(np.mean((lvis_counts_all - aop_counts_all) ** 2))
    mae = np.mean(np.abs(lvis_counts_all - aop_counts_all))
    if len(lvis_counts_all) > 1:
        correlation = np.corrcoef(lvis_counts_all, aop_counts_all)[0, 1]
    else:
        correlation = 0.0
    ss_res = np.sum((aop_counts_all - lvis_counts_all) ** 2)
    ss_tot = np.sum((aop_counts_all - np.mean(aop_counts_all)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Normalized metrics on concatenated normalized waveform pairs
    rmse_normalized = np.sqrt(np.mean((lvis_counts_norm_all - aop_counts_norm_all) ** 2))
    if len(lvis_counts_norm_all) > 1:
        correlation_normalized = np.corrcoef(lvis_counts_norm_all, aop_counts_norm_all)[0, 1]
    else:
        correlation_normalized = 0.0
    
    print("\n" + "="*60)
    print(f"Correlation between LVIS and AOP waveforms:")
    print("="*60)
    print(f"  Correlation: {correlation:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Normalized correlation: {correlation_normalized:.4f}")
    print(f"  Normalized RMSE: {rmse_normalized:.4f}")
    print(f"  R²: {r_squared:.4f}")
    print(f"  MAE: {mae:.4f}")
    print("="*60)

@torch.no_grad()
def predict_batch(model, dataloader, device='cpu', return_targets=True):
    """Predict for entire dataloader."""
    model.to(device)
    model.eval()
    
    all_mean_counts, all_mu, all_r = [], [], []
    all_targets, all_data_masks = [], []
    output_heights = None
    
    for batch in tqdm(dataloader, desc="Predicting"):
        lvis_counts = batch['lvis_counts'].to(device)
        lvis_heights = batch['lvis_heights'].to(device)
        lvis_padding_mask = batch['lvis_padding_mask'].to(device)
        
        output = model(
            photon_counts=lvis_counts,
            heights=lvis_heights,
            padding_mask=lvis_padding_mask
        )
        
        all_mean_counts.append(output['mean_counts'].cpu().numpy())
        all_mu.append(output['predictions']['mu'].cpu().numpy())
        all_r.append(output['predictions']['r'].cpu().numpy())
        
        if return_targets:
            all_targets.append(batch['als_target'].numpy())
            all_data_masks.append(batch['data_mask'].numpy())
        
        if output_heights is None:
            output_heights = output['output_heights'].cpu().numpy()
    
    results = {
        'heights': output_heights,
        'mean_counts': np.concatenate(all_mean_counts, axis=0),
        'mu': np.concatenate(all_mu, axis=0),
        'r': np.concatenate(all_r, axis=0),
    }
    results['std'] = np.sqrt(results['mu'] + results['mu']**2 / (results['r'] + 1e-8))
    
    if return_targets:
        results['targets'] = np.concatenate(all_targets, axis=0)
        results['data_masks'] = np.concatenate(all_data_masks, axis=0)
    
    return results

def get_FHD(wf, bin_width=1.0):
    """
    Compute FHD (Shannon index) for a waveform array.
    wf: Nx2 array-like where col0=heights, col1=counts
    bin_width: width of height bins (default 1.0 m)
    Returns: FHD (float)
    """
    wf = np.asarray(wf)
    if wf.size == 0:
        return 0.0

    heights = wf[:, 0].astype(float)
    counts = wf[:, 1].astype(float)

    # ensure heights ascending for digitize
    sort_idx = np.argsort(heights)
    heights = heights[sort_idx]
    counts = counts[sort_idx]

    # make bin edges from floor(min) to ceil(max) with step bin_width
    bin_edges = np.arange(np.floor(heights.min()),
                          np.ceil(heights.max()) + bin_width,
                          bin_width)

    # digitize and aggregate
    bin_idx = np.digitize(heights, bin_edges) - 1
    binned_counts = np.zeros(max(0, len(bin_edges) - 1))
    for i in range(len(binned_counts)):
        mask = (bin_idx == i)
        if mask.any():
            binned_counts[i] = counts[mask].sum()

    # keep same ordering as your original cell
    binned_counts = binned_counts[::-1]

    # Shannon index (ignore zero bins)
    nz = binned_counts[binned_counts > 0]
    if nz.size == 0:
        return 0.0
    p = nz / nz.sum()
    
    return -np.sum(p * np.log(p))

def get_VCR(wf):
    
    Compute variance (StdBin) of heights weighted by counts for a waveform array.
    wf: Nx2 array-like where column 0 = heights, column 1 = counts.
    Returns: StdBin (float)
    
    counts = np.asarray(wf[:, 1], dtype=float)
    heights = np.asarray(wf[:, 0], dtype=float)
    total_counts = counts.sum()
    if total_counts == 0:
        normalized_counts = counts  # all zeros -> variance zero
    else:
        normalized_counts = counts / total_counts

    H_x = (heights * normalized_counts).sum()
    StdBin = (normalized_counts * (heights - H_x) ** 2).sum()

    return StdBin

def evaluate_CSC_predictions(predictions, print_results=True):
    """
    Compute FHD and VCR metrics between predicted and target waveforms.
    Returns dict with mean absolute error and correlation for both metrics.
    """
    pred = predictions['mean_counts']
    target = predictions['targets']
    mask = predictions['data_masks']
    heights = predictions['heights']

    # Compute FHD and VCR for each sample (only on valid region)
    FHD_pred, FHD_target = [], []
    VCR_pred, VCR_target = [], []

    for i in tqdm(range(pred.shape[0])):
        valid = mask[i] > 0
        h = heights[valid]
        pred_counts = pred[i][valid]
        target_counts = target[i][valid]

        # Skip empty
        if len(h) == 0:
            continue

        # Stack as [height, count]
        pred_wf = np.stack([h, pred_counts], axis=1)
        target_wf = np.stack([h, target_counts], axis=1)

        FHD_pred.append(get_FHD(pred_wf))
        FHD_target.append(get_FHD(target_wf))
        VCR_pred.append(get_VCR(pred_wf))
        VCR_target.append(get_VCR(target_wf))

    FHD_pred = np.array(FHD_pred)
    FHD_target = np.array(FHD_target)
    VCR_pred = np.array(VCR_pred)
    VCR_target = np.array(VCR_target)

    metrics = {
        'FHD_rmse': np.sqrt(np.mean((FHD_pred - FHD_target) ** 2)),
        'FHD_corr': np.corrcoef(FHD_pred, FHD_target)[0, 1] if len(FHD_pred) > 1 else 0.0,
        'VCR_rmse': np.sqrt(np.mean((VCR_pred - VCR_target) ** 2)),
        'VCR_corr': np.corrcoef(VCR_pred, VCR_target)[0, 1] if len(VCR_pred) > 1 else 0.0,
    }

    if print_results:
        print("\n" + "="*60)
        print("CSC Evaluation Results:")
        print("="*60)
        print(f"  VCR Corr: {metrics['VCR_corr']:.4f}")        
        print(f"  VCR RMSE: {metrics['VCR_rmse']:.4f}")
        print(f"  FHD Corr: {metrics['FHD_corr']:.4f}")
        print(f"  FHD RMSE: {metrics['FHD_rmse']:.4f}")
        print("="*60)

    return metrics

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load data
    lvis_waveforms, als_waveforms = load_data(args)
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    config = checkpoint['config']

    # Create data loaders
    _, _, test_loader = create_data_splits(
        lvis_waveforms, als_waveforms, config
    )

    # Create model
    model = LVIS2ALSTransformer(
        embed_dim=config.embed_dim,
        lvis_height_range=config.lvis_height_range,
        als_height_range=config.als_height_range,
        height_resolution=config.height_resolution,
        global_max_count=config.global_max_count,
        global_max_sum=config.global_max_sum,
        use_auxiliary_features=config.use_auxiliary_features,
        use_cls_token=config.use_cls_token,
        als_query_fusion=config.als_query_fusion,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        encoder_heads=config.num_heads,
        decoder_heads=config.num_heads,
        encoder_ffn_dim=config.ffn_dim,
        decoder_ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        output_distribution="negative_binomial"
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    LVIS_wf_test = test_loader.dataset.lvis_waveforms
    AOP_wf_test = test_loader.dataset.als_waveforms

    # Waveform Correlation with LVIS baseline
    compute_lvis_aop_metrics(LVIS_wf_test, AOP_wf_test, config)

    # Lists to store FHD and VCR values
    LVIS_FHD_list = []
    LVIS_VCR_list = []
    AOP_FHD_list = []
    AOP_VCR_list = []

    # Compute metrics for LVIS test waveforms
    for wf in tqdm(LVIS_wf_test):
        LVIS_FHD_list.append(get_FHD(wf))
        LVIS_VCR_list.append(get_VCR(wf))

    # Compute metrics for AOP test waveforms
    for wf in tqdm(AOP_wf_test):
        AOP_FHD_list.append(get_FHD(wf))
        AOP_VCR_list.append(get_VCR(wf))

    # CSC correlation with LVIS baseline
    VCR_corr = np.corrcoef(AOP_VCR_list, LVIS_VCR_list)
    FHD_corr = np.corrcoef(AOP_FHD_list, LVIS_FHD_list)
    VCR_corr_value = VCR_corr[0, 1]
    FHD_corr_value = FHD_corr[0, 1]

    print("\n" + "="*60)
    print(f"CSC Metrics: LVIS vs Ground Truth ALS")
    print("="*60)
    print(f"VCR correlation: {VCR_corr_value:.4f}")
    print(f"FHD correlation: {FHD_corr_value:.4f}")
    print("="*60)

    # Generate predictions
    predictions = predict_batch(model, test_loader, device=config.device)

    # Evaluate CSC metrics
    csc_metrics = evaluate_CSC_predictions(predictions)

if __name__ == '__main__':
    main()
