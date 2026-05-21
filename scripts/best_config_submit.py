"""
Main training script for LVIS to ALS model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --batch_size 16 --epochs 50
"""

import argparse
import sys
from pathlib import Path
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from data.dataset import create_data_splits
from model import LVIS2ALSTransformer
from training.config import TrainingConfig
from training.trainer import Trainer
from utils.visualize import plot_training_history

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LVIS to ALS model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--lvis_file', type=str, default='LVIS_wf_list.pkl',
                       help='LVIS waveforms file')
    parser.add_argument('--als_file', type=str, default='AOP_wf_list.pkl',
                       help='ALS waveforms file')
    # parser.add_argument('--use_auxiliary_features', action='store_true',
    #                    help='Use auxiliary features')
    # parser.add_argument('--augment_data', action='store_true',
    #                    help='Apply data augmentation')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
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

def main():
    """Main training function."""
    args = parse_args()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load data
    lvis_waveforms, als_waveforms = load_data(args)
    
    # Configure training
    config = TrainingConfig(

        num_workers=0,  # Adjust based on your system
        # Model architecture
        embed_dim=128,
        use_auxiliary_features=True,
        use_cls_token=False,
        als_query_fusion="Add",
        encoder_layers=4,
        decoder_layers=4,
        num_heads=4,
        ffn_dim=512,
        dropout=0.2,
        
        # Your data statistics
        lvis_height_range=(-15.0, 82.0),
        als_height_range=(1.0, 80.0),
        height_resolution=0.15,
        global_max_count=4093.0,
        global_max_sum=80684.0,
        
        # Loss weights (tune based on validation)
        lambda_count=0.6,    # Count accuracy in data region
        lambda_zero_penalty=0.4,  # Zero prediction outside data region
        lambda_shape=1.5,         # Waveform shape similarity
        lambda_fhd=0.6,		# FHD loss weight
        lambda_vcr=0.0,		# VCR loss weight	 
        
        # Training parameters
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.05,
        num_epochs=100,
        scheduler="cosine",
        warmup_epochs=5,
        early_stopping_patience=15,
        
        # Data
        train_split=0.8,
        val_split=0.1,
        augment_train=False,
        boundary_buffer=2.0,

        #Checkpointing
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_splits(
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
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Continuing from epoch {start_epoch}")
    
    # Train
    results = trainer.train()
    
    # Test
    test_metrics = trainer.test()
    
    # Plot results
    plot_training_history(results, save_path=str(Path(config.checkpoint_dir) / 'training_curves.png'))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {min(results['val_loss']):.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²: {test_metrics['r_squared']:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
