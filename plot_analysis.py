import matplotlib.pyplot as plt
import matplotlib
import pickle
import json
import os
import numpy as np

matplotlib.use('Agg')

def plot_detailed_analysis(results_dict, output_dir='plots'):
    """
    Create detailed analysis plots from evaluation results.
    
    Args:
        results_dict: Dictionary with dataset names as keys and results as values
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    datasets = list(results_dict.keys())
    precisions = [results_dict[d]['precision'] for d in datasets]
    recalls = [results_dict[d]['recall'] for d in datasets]
    f1_scores = [results_dict[d]['f1'] for d in datasets]
    
    # Plot 1: Grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.25
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='lightcoral', edgecolor='black')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='lightgreen', edgecolor='black')
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Transformer Model Performance on Different Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/performance_comparison.png")
    plt.close()
    
    # Plot 2: Line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = range(len(datasets))
    
    ax.plot(x_pos, precisions, 'b-o', linewidth=2, markersize=8, label='Precision')
    ax.plot(x_pos, recalls, 'r-s', linewidth=2, markersize=8, label='Recall')
    ax.plot(x_pos, f1_scores, 'g-^', linewidth=2, markersize=8, label='F1 Score')
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Metrics Trend', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_trend.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/metrics_trend.png")
    plt.close()

def plot_algorithm_comparison(all_algorithms, output_dir='plots'):
    """
    Compare multiple algorithms on the same datasets.
    
    Args:
        all_algorithms: Dict with algorithm names as keys, and dicts of results as values
                       Format: {'FMM': {'PKU': {...}, 'MSR': {...}}, ...}
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    algorithms = list(all_algorithms.keys())
    datasets = list(all_algorithms[algorithms[0]].keys()) if algorithms else []
    
    # Extract F1 scores
    f1_by_dataset = {}
    for dataset in datasets:
        f1_by_dataset[dataset] = [all_algorithms[algo][dataset]['f1'] for algo in algorithms]
    
    # Plot: F1 Scores by Dataset
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algorithms))
    width = 0.2
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    for i, dataset in enumerate(datasets):
        offset = (i - len(datasets)/2 + 0.5) * width
        ax.bar(x + offset, f1_by_dataset[dataset], width, label=dataset, color=colors[i % len(colors)], edgecolor='black')
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Comparison (F1 Scores)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/algorithm_comparison.png")
    plt.close()
    
    # Plot: Average F1 Scores
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_f1 = {algo: np.mean([all_algorithms[algo][ds]['f1'] for ds in datasets]) for algo in algorithms}
    
    colors_algo = ['green', 'blue', 'orange', 'red']
    bars = ax.barh(algorithms, list(avg_f1.values()), color=colors_algo[:len(algorithms)], edgecolor='black')
    
    ax.set_xlabel('Average F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Average Performance', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (algo, score) in enumerate(avg_f1.items()):
        ax.text(score + 0.02, i, f'{score:.4f}', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/average_comparison.png")
    plt.close()

def main():
    """Example usage with sample data."""
    
    # Example Transformer results
    transformer_results = {
        'PKU': {'precision': 0.6330, 'recall': 0.7550, 'f1': 0.6886},
        'MSR': {'precision': 0.6789, 'recall': 0.8051, 'f1': 0.7366},
        'CITYU': {'precision': 0.6258, 'recall': 0.7640, 'f1': 0.6880},
        'AS': {'precision': 0.7363, 'recall': 0.8470, 'f1': 0.7877}
    }
    
    # Example comparison of all algorithms
    all_algorithms = {
        'FMM': {
            'PKU': {'precision': 0.8638, 'recall': 0.8480, 'f1': 0.8558},
            'MSR': {'precision': 0.8971, 'recall': 0.9081, 'f1': 0.9026},
            'CITYU': {'precision': 0.8449, 'recall': 0.8684, 'f1': 0.8565},
            'AS': {'precision': 0.8577, 'recall': 0.8754, 'f1': 0.8665}
        },
        'BMM': {
            'PKU': {'precision': 0.8654, 'recall': 0.8496, 'f1': 0.8574},
            'MSR': {'precision': 0.8994, 'recall': 0.9101, 'f1': 0.9047},
            'CITYU': {'precision': 0.8474, 'recall': 0.8709, 'f1': 0.8590},
            'AS': {'precision': 0.8574, 'recall': 0.8752, 'f1': 0.8662}
        },
        'HMM': {
            'PKU': {'precision': 0.7911, 'recall': 0.7761, 'f1': 0.7835},
            'MSR': {'precision': 0.7716, 'recall': 0.7967, 'f1': 0.7840},
            'CITYU': {'precision': 0.7546, 'recall': 0.7571, 'f1': 0.7559},
            'AS': {'precision': 0.7914, 'recall': 0.7964, 'f1': 0.7939}
        },
        'Transformer': transformer_results
    }
    
    print("Generating analysis plots...")
    print("-" * 50)
    
    # Plot Transformer detailed analysis
    print("\n1. Transformer Detailed Analysis")
    plot_detailed_analysis(transformer_results, output_dir='plots')
    
    # Plot algorithm comparison
    print("\n2. Algorithm Comparison")
    plot_algorithm_comparison(all_algorithms, output_dir='plots')
    
    print("\n" + "="*50)
    print("All plots saved to 'plots' directory!")
    print("="*50)

if __name__ == "__main__":
    main()
