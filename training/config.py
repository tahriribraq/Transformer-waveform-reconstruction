"""
Training configuration.
"""

import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Data
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split is implicitly 1 - train_split - val_split

    # Embedding
    use_auxiliary_features: bool = True
    use_cls_token: bool = True
    als_query_fusion: str = "Add"  # "Add" or "MLP"
    
    # Model
    embed_dim: int = 256
    encoder_layers: int = 6
    decoder_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    
    # Height ranges
    lvis_height_range: Tuple[float, float] = (-15.0, 85.0)
    als_height_range: Tuple[float, float] = (0.0, 100.0)
    height_resolution: float = 0.15
    
    # Data statistics
    global_max_count: float = 4096.0
    global_max_sum: float = 80684.0

    # Augmentation
    augment_train: bool = False
    boundary_buffer: float = 2.0
    
    # Loss weights
    lambda_count: float = 1.0
    lambda_shape: float = 0.3
    lambda_zero_penalty: float = 1.0
    lambda_fhd: float = 0.6
    lambda_vcr: float = 0.5
    lambda_emd: float = 0.0
    lambda_peak: float = 0.0
    
    # FHD settings
    bin_width: float = 1.0  # Match evaluation
    use_correlation: bool = False  # MSE vs correlation loss
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "onecycle"
    warmup_epochs: int = 5
    
    # Training
    num_epochs: int = 100
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    seed: int = 42
