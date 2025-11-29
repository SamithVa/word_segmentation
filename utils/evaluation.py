"""
Evaluation metrics for word segmentation.
"""

from tqdm import tqdm
import os
import pandas as pd
from tabulate import tabulate
from config import OUTPUT_DIR, RESULTS_DIR


def evaluate_segmentation(tokenizer, gold_filepath, test_filepath):
    """
    Evaluate word segmentation performance.
    
    Args:
        tokenizer: tokenizer instance with tokenize() method
        gold_filepath: path to gold standard file
        test_filepath: path to test file (unsegmented)
        
    Returns:
        dict with precision, recall, and f1 scores
    """
    # Read test and gold data
    with open(test_filepath, 'r', encoding='utf-8') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    with open(gold_filepath, 'r', encoding='utf-8') as f:
        gold_lines = [line.strip() for line in f if line.strip()]
    
    min_len = min(len(test_lines), len(gold_lines))
    test_lines = test_lines[:min_len]
    gold_lines = gold_lines[:min_len]
    
    total_pred_words = 0
    total_gold_words = 0
    total_correct_words = 0
    
    for test_line, gold_line in tqdm(zip(test_lines, gold_lines), total=len(test_lines), desc="Evaluating"):
        pred_words = tokenizer.tokenize(test_line)
        gold_words = gold_line.split()
        
        total_pred_words += len(pred_words)
        total_gold_words += len(gold_words)
        
        # Calculate correct words using position matching
        pred_positions = []
        pos = 0
        for word in pred_words:
            pred_positions.append((pos, pos + len(word), word))
            pos += len(word)
        
        gold_positions = []
        pos = 0
        for word in gold_words:
            gold_positions.append((pos, pos + len(word), word))
            pos += len(word)
        
        # Count matches
        for pred_pos in pred_positions:
            if pred_pos in gold_positions:
                total_correct_words += 1
    
    precision = total_correct_words / total_pred_words if total_pred_words > 0 else 0
    recall = total_correct_words / total_gold_words if total_gold_words > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_model_on_all_datasets(model, datasets, model_name="Model", output_dir=None):
    """
    Evaluate a model on all datasets and display comprehensive results.

    Args:
        model: The model/tokenizer to evaluate (must have tokenize() method)
        datasets: Dictionary of datasets (from config.DATASETS)
        model_name: Name of the model for display purposes
        output_dir: Directory to save results (defaults to RESULTS_DIR)

    Returns:
        Dictionary containing all results
    """
    # Use RESULTS_DIR as default if not specified
    if output_dir is None:
        output_dir = RESULTS_DIR

    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on All Datasets")
    print(f"{'='*60}")

    all_results = {}

    for dataset_name, dataset_paths in datasets.items():
        gold_path = dataset_paths['gold']
        test_path = dataset_paths['test']

        if os.path.exists(gold_path) and os.path.exists(test_path):
            print(f"\n--- Evaluating {dataset_name.upper()} ---")
            try:
                results = evaluate_segmentation(model, gold_path, test_path)
                all_results[dataset_name.upper()] = results
                print(f"Precision: {results['precision']:.4f}")
                print(f"Recall:    {results['recall']:.4f}")
                print(f"F1:         {results['f1']:.4f}")
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
        else:
            print(f"Skipping {dataset_name}: files not found")

    if all_results:
        # Display summary table
        print(f"\n{'='*60}")
        print(f"Summary for {model_name}")
        print(f"{'='*60}")
        print(f"{'Dataset':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 50)

        for name, results in all_results.items():
            print(f"{name:<10} {results['precision']:<12.4f} {results['recall']:<12.4f} {results['f1']:<12.4f}")

        # Calculate and display average scores
        avg_p = sum(r['precision'] for r in all_results.values()) / len(all_results)
        avg_r = sum(r['recall'] for r in all_results.values()) / len(all_results)
        avg_f1 = sum(r['f1'] for r in all_results.values()) / len(all_results)
        print("-" * 50)
        print(f"{'Average':<10} {avg_p:<12.4f} {avg_r:<12.4f} {avg_f1:<12.4f}")

        # Save results to CSV
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame([
            {
                'Dataset': name,
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1': f"{results['f1']:.4f}"
            }
            for name, results in all_results.items()
        ])
        output_file = os.path.join(output_dir, f'{model_name.lower()}_evaluation_results.csv')
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    return all_results
