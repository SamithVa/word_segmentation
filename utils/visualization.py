"""
Visualization utilities for training metrics and model comparison.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use non-interactive backend


def plot_training_history(train_losses, train_accuracies, output_path='training_history.png', model_name='Model'):
    """
    Plot training loss and accuracy curves with simple, clean visualization.

    Args:
        train_losses: list of training losses per epoch
        train_accuracies: list of training accuracies per epoch
        output_path: path to save plot
        model_name: name of the model for display purposes
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    epochs = list(range(1, len(train_losses) + 1))

    # Plot loss
    ax1.plot(epochs, train_losses, 'o-', linewidth=2, markersize=6, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add loss values with better spacing
    for i, loss in enumerate(train_losses):
        ax1.text(epochs[i], loss, f'{loss:.4f}', ha='center', va='bottom',
                 fontsize=9, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

    # Plot accuracy
    ax2.plot(epochs, train_accuracies, 's-', linewidth=2, markersize=6, label='Training Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{model_name} Training Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Add accuracy values with better spacing
    for i, acc in enumerate(train_accuracies):
        ax2.text(epochs[i], acc, f'{acc:.4f}', ha='center', va='top',
                 fontsize=9, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {output_path}")
    plt.close()


def plot_model_comparison(models, precision, recall, f1_scores, output_path='model_comparison.png'):
    """
    Create a simple bar chart comparing model performance.

    Args:
        models: List of model names
        precision: List of precision scores
        recall: List of recall scores
        f1_scores: List of F1 scores
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(models))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)

    # Customize
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Chinese Word Segmentation: Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.6, 1.0)

    # Add value labels with better visibility
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            # Add background box for better visibility
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2., height),
                       xytext=(0, 8),  # 8 points vertical offset
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {output_path}")
    plt.close()