"""
Dataset for LVIS-ALS waveform pairs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
#from lvis2als.training.config import TrainingConfig
from training.config import TrainingConfig

class LVISALSDataset(Dataset):
    """
    Dataset for LVIS-ALS waveform pairs.
    
    Handles:
    - Variable length LVIS sequences
    - Conversion of ALS to fixed output grid
    - Creation of data_mask for loss computation
    - Optional data augmentation
    """
    
    def __init__(
        self,
        lvis_waveforms: List[np.ndarray],
        als_waveforms: List[np.ndarray],
        als_height_range: Tuple[float, float] = (0.0, 100.0),
        height_resolution: float = 0.15,
        augment: bool = False,
        boundary_buffer: float = 2.0
    ):
        """
        Args:
            lvis_waveforms: List of [seq_len, 2] arrays (height, count)
            als_waveforms: List of [seq_len, 2] arrays (height, count)
            als_height_range: Output height range for ALS
            height_resolution: Height bin resolution in meters
            augment: Whether to apply data augmentation
            boundary_buffer: Buffer around ALS data range (meters)
        """
        assert len(lvis_waveforms) == len(als_waveforms)
        
        self.lvis_waveforms = lvis_waveforms
        self.als_waveforms = als_waveforms
        self.als_height_range = als_height_range
        self.height_resolution = height_resolution
        self.augment = augment
        self.boundary_buffer = boundary_buffer

        # Calculate number of output bins
        self.num_output_bins = int(
            (als_height_range[1] - als_height_range[0]) / height_resolution
        ) + 1
        
        # Pre-compute output height grid
        self.output_heights = np.linspace(
            als_height_range[0],
            als_height_range[1],
            self.num_output_bins
        )
    
    def __len__(self) -> int:
        return len(self.lvis_waveforms)
    
    def _als_to_fixed_grid(self, als_wf: np.ndarray) -> np.ndarray:
        """Convert variable-length ALS waveform to fixed output grid.
        Args:
            als_wf: [seq_len, 2] array with (height, count)
            
        Returns:
            [num_output_bins] array of counts on fixed grid
        """
        output = np.zeros(self.num_output_bins, dtype=np.float32)
        
        if len(als_wf) == 0:
            return output
        
        heights = als_wf[:, 0]
        counts = als_wf[:, 1]
        
        for h, c in zip(heights, counts):
            if self.als_height_range[0] <= h <= self.als_height_range[1]:
                idx = int((h - self.als_height_range[0]) / self.height_resolution)
                idx = min(idx, self.num_output_bins - 1)
                output[idx] += c
        
        return output
    
    def _create_data_mask(self, als_wf: np.ndarray) -> np.ndarray:
        """
        Create binary mask indicating the valid data region.
        
        Args:
            als_wf: [seq_len, 2] array with (height, count)   
        Returns:
            [num_output_bins] mask (1.0 for data region, 0.0 for zero region)
        """
        data_mask = np.zeros(self.num_output_bins, dtype=np.float32)
        
        if len(als_wf) == 0:
            # No ALS data - entire output should be zeros
            return data_mask
        
        # Get height range of actual ALS data
        als_h_min = als_wf[:, 0].min()
        als_h_max = als_wf[:, 0].max()
        
        # Add buffer for boundary effects
        valid_min = als_h_min - self.boundary_buffer
        valid_max = als_h_max + self.boundary_buffer
        
        # Mark positions within valid range
        for i, h in enumerate(self.output_heights):
            if valid_min <= h <= valid_max:
                data_mask[i] = 1.0
        
        return data_mask
    
    def _augment(
        self,
        lvis_heights: np.ndarray,
        lvis_counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to LVIS waveform.
        
        Augmentations:
        - Height jitter (simulates registration errors)
        - Count scaling (simulates intensity variations)
        - Additive noise
        """
        if not self.augment:
            return lvis_heights, lvis_counts
        
        # Height jitter: ±0.3m
        if np.random.rand() < 0.3:
            jitter = np.random.uniform(-0.3, 0.3)
            lvis_heights = lvis_heights + jitter
        
        # Count scaling: 0.8x to 1.2x
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            lvis_counts = lvis_counts * scale
        
        # Additive noise
        if np.random.rand() < 0.2:
            noise_std = lvis_counts.mean() * 0.05
            noise = np.random.normal(0, noise_std, lvis_counts.shape)
            lvis_counts = np.maximum(0, lvis_counts + noise)
        
        return lvis_heights, lvis_counts
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lvis_wf = self.lvis_waveforms[idx]
        als_wf = self.als_waveforms[idx]

        # Extract LVIS data
        lvis_heights = lvis_wf[:, 0].astype(np.float32)
        lvis_counts = lvis_wf[:, 1].astype(np.float32) 
        # Apply augmentation
        lvis_heights, lvis_counts = self._augment(lvis_heights, lvis_counts)
        # Convert ALS to fixed grid
        als_target = self._als_to_fixed_grid(als_wf)
        # Create data mask
        data_mask = self._create_data_mask(als_wf)
        
        return {
            'lvis_counts': torch.from_numpy(lvis_counts),
            'lvis_heights': torch.from_numpy(lvis_heights),
            'als_target': torch.from_numpy(als_target),
            'data_mask': torch.from_numpy(data_mask),
            'lvis_length': len(lvis_counts)
        }
    
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Handles variable-length LVIS sequences with padding.
    """
    # Find max LVIS length in batch
    max_lvis_len = max(item['lvis_length'] for item in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors for LVIS
    lvis_counts = torch.zeros(batch_size, max_lvis_len)
    lvis_heights = torch.zeros(batch_size, max_lvis_len)
    lvis_padding_mask = torch.ones(batch_size, max_lvis_len, dtype=torch.bool)
    
    # Stack fixed-size ALS data
    als_target = torch.stack([item['als_target'] for item in batch])
    data_mask = torch.stack([item['data_mask'] for item in batch])
    
    # Fill in LVIS data
    for i, item in enumerate(batch):
        length = item['lvis_length']
        lvis_counts[i, :length] = item['lvis_counts']
        lvis_heights[i, :length] = item['lvis_heights']
        lvis_padding_mask[i, :length] = False
    
    return {
        'lvis_counts': lvis_counts,
        'lvis_heights': lvis_heights,
        'lvis_padding_mask': lvis_padding_mask,
        'als_target': als_target,
        'data_mask': data_mask
    }

def create_data_splits(
    lvis_waveforms: List[np.ndarray],
    als_waveforms: List[np.ndarray],
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data splits and DataLoaders.
    
    Args:
        lvis_waveforms: List of [seq_len, 2] arrays (height, count)
        als_waveforms: List of [seq_len, 2] arrays (height, count)
        config: Training configuration
        
    Returns:
        train_loader, val_loader, test_loader
    """
    np.random.seed(config.seed)
    
    n_samples = len(lvis_waveforms)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * config.train_split)
    n_val = int(n_samples * config.val_split)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    print(f"Data splits: train={len(train_indices)}, "
          f"val={len(val_indices)}, test={len(test_indices)}")
    
    # Create datasets
    train_dataset = LVISALSDataset(
        lvis_waveforms=[lvis_waveforms[i] for i in train_indices],
        als_waveforms=[als_waveforms[i] for i in train_indices],
        als_height_range=config.als_height_range,
        height_resolution=config.height_resolution,
        augment=config.augment_train,
        boundary_buffer=config.boundary_buffer
    )
    
    val_dataset = LVISALSDataset(
        lvis_waveforms=[lvis_waveforms[i] for i in val_indices],
        als_waveforms=[als_waveforms[i] for i in val_indices],
        als_height_range=config.als_height_range,
        height_resolution=config.height_resolution,
        augment=False,
        boundary_buffer=config.boundary_buffer
    )
    
    test_dataset = LVISALSDataset(
        lvis_waveforms=[lvis_waveforms[i] for i in test_indices],
        als_waveforms=[als_waveforms[i] for i in test_indices],
        als_height_range=config.als_height_range,
        height_resolution=config.height_resolution,
        augment=False,
        boundary_buffer=config.boundary_buffer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.device == "cuda" else False
    )
    
    return train_loader, val_loader, test_loader