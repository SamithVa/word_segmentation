"""
HMM Chinese Word Segmentation - Training and Evaluation Script

This script provides functionality to:
1. Train a Hidden Markov Model (HMM) for Chinese word segmentation
2. Evaluate the trained model on SIGHAN Bakeoff 2005 datasets
3. Save and load trained models

Usage:
    # Train and evaluate
    python scripts/hmm.py

    # Only evaluate existing model
    python scripts/hmm.py --eval

    # Train with custom parameters
    python scripts/hmm.py --smoothing 1e-4 --no-preprocess
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hmm_seg import HMMSeg as HMMTokenizer
from utils.evaluation import evaluate_model_on_all_datasets
from config import DATASETS, HMM_SMOOTHING, HMM_USE_PREPROCESS, OUTPUT_DIR, RESULTS_DIR

def train(model_path, smoothing=HMM_SMOOTHING, use_preprocess=HMM_USE_PREPROCESS):
    """Train HMM model with configuration."""
    print("Training HMM model...")
    print(f"Smoothing: {smoothing}, Preprocessing: {use_preprocess}")

    # Get training files from config
    training_files = [DATASETS[dataset]['train'] for dataset in DATASETS.keys()]

    # Check which files exist
    existing_files = [f for f in training_files if os.path.exists(f)]
    print(f"Found {len(existing_files)} training file(s)")

    if not existing_files:
        print("No training files found!")
        return None

    # Create output directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Train model
    hmm = HMMTokenizer(smoothing=smoothing, use_preprocess=use_preprocess)
    hmm.train(existing_files)
    hmm.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Test on examples
    print("\n" + "="*50)
    print("Testing on Example Sentences")
    print("="*50)

    examples = [
        "即将来临时",
        "他在很多方面追求和平等",
        "研究生命起源"
    ]

    for sentence in examples:
        tokens = hmm.tokenize(sentence)
        print(f"Input:  {sentence}")
        print(f"Output: {' / '.join(tokens)}\n")

    return hmm

def evaluate(model_path, output_dir=RESULTS_DIR):
    """Evaluate HMM model using generic evaluation function."""
    print(f"Loading HMM model from {model_path}...")
    hmm = HMMTokenizer()
    hmm.load_model(model_path)

    return evaluate_model_on_all_datasets(
        model=hmm,
        datasets=DATASETS,
        model_name="HMM",
        output_dir=output_dir
    )

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='HMM Chinese Word Segmentation')
    parser.add_argument('--eval', action='store_true',
                        help='Only evaluate existing model')
    parser.add_argument('--model-path', type=str,
                        default=os.path.join(OUTPUT_DIR, 'saved_models', 'hmm_model.pkl'),
                        help='Path to save/load model')
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR,
                        help='Directory to save evaluation outputs')
    parser.add_argument('--smoothing', type=float, default=HMM_SMOOTHING,
                        help=f'Smoothing parameter for HMM (default: {HMM_SMOOTHING})')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Disable preprocessing (number/English normalization)')

    args = parser.parse_args()

    # Set parameters
    model_path = args.model_path
    output_dir = args.output_dir
    smoothing = args.smoothing
    use_preprocess = not args.no_preprocess

    # Run based on mode
    if args.eval:
        print("="*60)
        print("HMM Evaluation Mode")
        print("="*60)
        evaluate(model_path, output_dir)
    else:  # train by default
        print("="*60)
        print("HMM Training and Evaluation Mode")
        print("="*60)
        hmm = train(model_path, smoothing, use_preprocess)
        if hmm:  # Only evaluate if training succeeded
            evaluate(model_path, output_dir)

if __name__ == "__main__":
    main()