"""
Trainer for LVIS2ALS model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from pathlib import Path
import time
from typing import Tuple, Dict, Optional
from tqdm import tqdm

from .config import TrainingConfig
from .losses import CombinedLoss
from .metrics import Metrics
import numpy as np

class Trainer:
    """
    Trainer for LVIS2ALS model.
    
    Handles:
    - Training loop with gradient clipping
    - Validation with metrics computation
    - Checkpointing (best, latest, periodic)
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = config.device
        self.start_epoch = 1
        
        # Compute output heights for FHD loss
        num_output_bins = int(
            (config.als_height_range[1] - config.als_height_range[0]) / config.height_resolution
        ) + 1
        output_heights = np.linspace(
            config.als_height_range[0],
            config.als_height_range[1],
            num_output_bins
        )
        
        # Loss function
        self.criterion = CombinedLoss(
            output_heights=output_heights,
            lambda_data_count=config.lambda_count,
            lambda_zero_penalty=config.lambda_zero_penalty,
            lambda_shape=config.lambda_shape,
            lambda_peak=config.lambda_peak,
            lambda_emd=config.lambda_emd,
            lambda_fhd=config.lambda_fhd,
            lambda_vcr=config.lambda_vcr,
            bin_width=1.0,  # Match your evaluation function
            use_correlation=False  # Start with MSE, try correlation if needed
        ).to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay, # L2 regularization
            betas=config.betas # Adam momentum parameters
        )
        
        # Scheduler
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, config.num_epochs // 4),
                T_mult=2,
                eta_min=config.learning_rate * 0.01
            )
        else:  # onecycle
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                epochs=config.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=config.warmup_epochs / config.num_epochs
            )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_correlation = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_losses': [],  # Detailed loss components
            'val_losses': [],
            'val_metrics': []
        }
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Store output heights for metrics
        self.output_heights = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        loss_keys = ['total', 'data_count', 'shape', 'zero_penalty', 'fhd', 'vcr', 'emd', 'peak']
        total_losses = {k: 0.0 for k in loss_keys}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for i, batch in enumerate(pbar):
            # Move to device
            lvis_counts = batch['lvis_counts'].to(self.device)
            lvis_heights = batch['lvis_heights'].to(self.device)
            lvis_padding_mask = batch['lvis_padding_mask'].to(self.device)
            als_target = batch['als_target'].to(self.device)
            data_mask = batch['data_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad() # Clear previous gradients
            
            output = self.model(
                photon_counts=lvis_counts,
                heights=lvis_heights,
                padding_mask=lvis_padding_mask
            )
            
            # Compute loss
            losses = self.criterion(
                predictions=output['predictions'],
                target=als_target,
                data_mask=data_mask
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
            
            self.optimizer.step()
            
            # Update scheduler for OneCycleLR
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Track losses
            for key in loss_keys:
                total_losses[key] += losses[key].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'count': f"{losses['data_count'].item():.4f}",
                'shape': f"{losses['shape'].item():.4f}",
                'zero': f"{losses['zero_penalty'].item():.4f}",
                'fhd': f"{losses['fhd'].item():.4f}",
                'vcr': f"{losses['vcr'].item():.4f}",
                'emd': f"{losses['emd'].item():.4f}",
                'peak': f"{losses['peak'].item():.4f}"
            })
            
            # Memory monitoring every 100 batches
            # if i % 100 == 0:
            #     current, peak = tracemalloc.get_traced_memory()
            #     print(f"Current memory: {current / 1e9:.2f}GB, Peak: {peak / 1e9:.2f}GB")
        
        # Average losses
        avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
        
        # Update scheduler for CosineAnnealing
        if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step()
        
        return avg_losses
    
    @torch.no_grad() # Disable gradient computation (faster, less memory)
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate the model."""
        self.model.eval() # Disable dropout, use batchnorm running stats
        
        loss_keys = ['total', 'data_count', 'shape', 'zero_penalty', 'fhd', 'vcr', 'emd', 'peak']
        total_losses = {k: 0.0 for k in loss_keys}
        num_batches = 0
        
        all_preds = []
        all_targets = []
        all_data_masks = []
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            lvis_counts = batch['lvis_counts'].to(self.device)
            lvis_heights = batch['lvis_heights'].to(self.device)
            lvis_padding_mask = batch['lvis_padding_mask'].to(self.device)
            als_target = batch['als_target'].to(self.device)
            data_mask = batch['data_mask'].to(self.device)
            
            output = self.model(
                photon_counts=lvis_counts,
                heights=lvis_heights,
                padding_mask=lvis_padding_mask
            )
            
            losses = self.criterion(
                predictions=output['predictions'],
                target=als_target,
                data_mask=data_mask
            )
            
            for key in loss_keys:
                total_losses[key] += losses[key].item()
            num_batches += 1
            
            # Collect predictions for metrics
            all_preds.append(output['mean_counts'].cpu())
            all_targets.append(als_target.cpu())
            all_data_masks.append(data_mask.cpu())
            
            # Store output heights (same for all batches)
            if self.output_heights is None:
                self.output_heights = output['output_heights'].cpu()
        
        # Average losses
        avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
        
        # Concatenate predictions
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_data_masks = torch.cat(all_data_masks, dim=0)
        
        # Compute metrics
        metrics = Metrics.compute_all(
            all_preds, all_targets, all_data_masks, self.output_heights
        )
        
        return avg_losses, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(), # Model weights
            'optimizer_state_dict': self.optimizer.state_dict(), # Optimizer state
            'scheduler_state_dict': self.scheduler.state_dict(), # LR schedule
            'best_val_loss': self.best_val_loss,
            'best_val_correlation': self.best_val_correlation,
            'config': self.config,
            'history': self.history
        }
        
        path = Path(self.config.checkpoint_dir)
        
        # Save latest
        torch.save(checkpoint, path / 'latest.pt')
        
        # Save periodic
        if epoch % self.config.save_every == 0:
            torch.save(checkpoint, path / f'epoch_{epoch}.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, path / 'best.pt')
    
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_correlation = checkpoint['best_val_correlation']
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch'] + 1
        
        return checkpoint['epoch']
    
    def train(self) -> Dict:
        """Full training loop."""
        print(f"\n{'=' * 60}")
        print("Training Configuration")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"{'=' * 60}\n")
        
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses, val_metrics = self.validate()
            
            # Track history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_losses'].append(train_losses)
            self.history['val_losses'].append(val_losses)
            self.history['val_metrics'].append(val_metrics)
            
            # Check for improvement
            #is_best = val_losses['total'] < self.best_val_loss
            is_best = val_metrics['correlation'] > self.best_val_correlation
            if is_best:
                self.best_val_correlation = val_metrics['correlation']
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{self.config.num_epochs} ({elapsed:.1f}s)")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(count={train_losses['data_count']:.4f}, "
                  f"shape={train_losses['shape']:.4f}, "
                  f"zero={train_losses['zero_penalty']:.4f}, "
                  f"fhd={train_losses['fhd']:.4f}, "
                  f"vcr={train_losses['vcr']:.4f}, "
                  f"emd={train_losses['emd']:.4f}, "
                  f"peak alignment={train_losses['peak']:.4f})")
            print(f"  Val Loss:   {val_losses['total']:.4f} "
                  f"(count={val_losses['data_count']:.4f}, "
                  f"shape={val_losses['shape']:.4f}, "
                  f"zero={val_losses['zero_penalty']:.4f}, "
                  f"fhd={train_losses['fhd']:.4f}, "
                  f"vcr={train_losses['vcr']:.4f}, "
                  f"emd={val_losses['emd']:.4f}, "
                  f"peak alignment={val_losses['peak']:.4f})")
            print(f"  Val Metrics: Corr={val_metrics['correlation']:.4f}, "
                  f"R²={val_metrics['r_squared']:.4f}, "
                  f"RMSE={val_metrics['rmse']:.2f}, "
                  f"ZeroAcc={val_metrics['zero_accuracy']:.2%}m")
          
            if is_best:
                print("  ★ New best model!")
            
            print()
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs "
                      f"(patience={self.config.early_stopping_patience})")
                break
        
        print(f"{'=' * 60}")
        print("Training Complete!")
        #print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        print(f"{'=' * 60}")
        
        return self.history
    
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        if self.test_loader is None:
            print("No test loader provided!")
            return {}
        
        # Load best model
        best_path = Path(self.config.checkpoint_dir) / 'best.pt'
        if best_path.exists():
            self.load_checkpoint(str(best_path))
            print("Loaded best model for testing (correlation: {checkpoint['best_val_correlation']:.4f})")
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_data_masks = []
        
        for batch in tqdm(self.test_loader, desc="Testing"):
            lvis_counts = batch['lvis_counts'].to(self.device)
            lvis_heights = batch['lvis_heights'].to(self.device)
            lvis_padding_mask = batch['lvis_padding_mask'].to(self.device)
            als_target = batch['als_target'].to(self.device)
            data_mask = batch['data_mask'].to(self.device)
            
            output = self.model(
                photon_counts=lvis_counts,
                heights=lvis_heights,
                padding_mask=lvis_padding_mask
            )
            
            all_preds.append(output['mean_counts'].cpu())
            all_targets.append(als_target.cpu())
            all_data_masks.append(data_mask.cpu())
            
            if self.output_heights is None:
                self.output_heights = output['output_heights'].cpu()
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_data_masks = torch.cat(all_data_masks, dim=0)
        
        metrics = Metrics.compute_all(
            all_preds, all_targets, all_data_masks, self.output_heights
        )
        
        print(f"\n{'=' * 60}")
        print("Test Results")
        print(f"{'=' * 60}")
        print(f"  Correlation:     {metrics['correlation']:.4f}")
        print(f"  RMSE:            {metrics['rmse']:.4f}")
        print(f"  Normalized correlation:     {metrics['n_correlation']:.4f}")
        print(f"  Normalized RMSE:            {metrics['n_rmse']:.4f}")      
        print(f"  R²:              {metrics['r_squared']:.4f}")
        print(f"  MAE:             {metrics['mae']:.4f}")
        print(f"  Zero Accuracy:   {metrics['zero_accuracy']:.2%}")
        print(f"{'=' * 60}")
        
        return metrics
