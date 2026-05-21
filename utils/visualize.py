"""
Visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history, save_path='training_curves.png'):
    """Plot training and validation curves."""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax1.axvline(best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
    ax1.plot(best_epoch, best_val_loss, 'g*', markersize=15)
    
    # Plot 2: Loss Components
    ax2 = axes[0, 1]
    
    # Extract components
    val_data_count = [x['data_count'] for x in history['val_losses']]
    val_shape = [x['shape'] for x in history['val_losses']]
    val_zero_penalty = [x['zero_penalty'] for x in history['val_losses']]
    
    ax2.plot(epochs, val_data_count, label='Data Count Loss', linewidth=2)
    ax2.plot(epochs, val_shape, label='Shape Loss', linewidth=2)
    ax2.plot(epochs, val_zero_penalty, label='Zero Penalty', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Component')
    ax2.set_title('Validation Loss Components')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Validation Metrics
    ax3 = axes[1, 0]
    
    rmse = [x['rmse'] for x in history['val_metrics']]
    r_squared = [x['r_squared'] for x in history['val_metrics']]
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(epochs, rmse, 'b-', label='RMSE', linewidth=2)
    line2 = ax3_twin.plot(epochs, r_squared, 'r-', label='R²', linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('RMSE', color='b')
    ax3_twin.set_ylabel('R²', color='r')
    ax3.set_title('Validation Metrics')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Correlation and Zero Accuracy
    ax4 = axes[1, 1]
    
    correlation = [x['correlation'] for x in history['val_metrics']]
    zero_accuracy = [x['zero_accuracy'] for x in history['val_metrics']]
    
    ax4.plot(epochs, correlation, 'g-', label='Correlation', linewidth=2)
    ax4.plot(epochs, zero_accuracy, 'purple', label='Zero Accuracy', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.set_title('Validation Correlation & Zero Accuracy')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.show()

def plot_predictions(predictions, num_samples=10, random_seed=42):
    """
    Plot predicted and target waveforms side-by-side for random samples.
    """
    import matplotlib.pyplot as plt
    np.random.seed(random_seed)
    pred = predictions['mean_counts']
    target = predictions['targets']
    mask = predictions['data_masks']
    heights = predictions['heights']

    n = pred.shape[0]
    idxs = np.random.choice(n, size=min(num_samples, n), replace=False)

    plt.figure(figsize=(15, 3 * len(idxs)))
    for i, idx in enumerate(idxs):
        valid = mask[idx] > 0
        h = heights[valid]
        pred_counts = pred[idx][valid]
        target_counts = target[idx][valid]

        plt.subplot(len(idxs), 2, 2 * i + 2)
        plt.plot(h, pred_counts, label='Prediction', color='orange')
        #plt.title("Prediction")
        plt.xlabel("Height (m)")
        plt.ylabel("ALS Counts")
        plt.grid(True)

        plt.subplot(len(idxs), 2, 2 * i + 1)
        plt.plot(h, target_counts, label='Target', color='blue')
        #plt.title("Target")
        plt.xlabel("Height (m)")
        plt.ylabel("ALS Counts")
        plt.grid(True)

    plt.tight_layout()
    plt.show()