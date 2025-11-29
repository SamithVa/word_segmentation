"""Test and evaluate classical Chinese word segmentation methods"""

import sys
import os
import pandas as pd
from tabulate import tabulate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FMM, BMM, BiMM
from config import DATASETS, RESULTS_DIR
from utils.evaluation import evaluate_segmentation, evaluate_model_on_all_datasets


def test_single_example():
    """Test methods on a single example"""
    test_text = "即将来临时他在很多方面追求和平等"

    print("Testing Chinese Word Segmentation Classical Methods")
    print("=" * 50)
    print(f"Test text: {test_text}")
    print()

    # Test classical methods
    print("Classical Methods:")
    print("-" * 30)

    # FMM
    fmm = FMM()
    fmm.load_dict([DATASETS['pku']['train']])
    fmm_result = fmm.tokenize(test_text)
    print(f"FMM:   {' / '.join(fmm_result)}")

    # BMM
    bmm = BMM()
    bmm.load_dict([DATASETS['pku']['train']])
    bmm_result = bmm.tokenize(test_text)
    print(f"BMM:   {' / '.join(bmm_result)}")

    # BiMM
    bimm = BiMM(dict_paths=[DATASETS['pku']['train']])
    bimm_result = bimm.tokenize(test_text)
    print(f"BiMM:  {' / '.join(bimm_result)}")

    print("\n" + "=" * 50 + "\n")


def evaluate_all_datasets():
    """Evaluate all classical methods on all datasets"""
    # Methods to evaluate
    methods = {
        'FMM': FMM,
        'BMM': BMM,
        'BiMM': BiMM
    }

    # Results storage
    all_results = []

    print("Evaluating Classical Chinese Word Segmentation Methods")
    print("=" * 60)

    for method_name, MethodClass in methods.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {method_name} Method")
        print(f"{'='*60}")

        # Initialize model with all datasets' training data
        dict_paths = [DATASETS[dataset]['train'] for dataset in DATASETS.keys()]

        if method_name == 'BiMM':
            model = MethodClass(dict_paths=dict_paths)
        else:
            model = MethodClass()
            model.load_dict(dict_paths)

        # Use generic evaluation function
        results = evaluate_model_on_all_datasets(
            model=model,
            datasets=DATASETS,
            model_name=method_name,
            output_dir=RESULTS_DIR
        )

        # Convert results format for comparison table
        for dataset, scores in results.items():
            all_results.append({
                'Dataset': dataset,
                'Method': method_name,
                'Precision': f"{scores['precision']:.4f}",
                'Recall': f"{scores['recall']:.4f}",
                'F1': f"{scores['f1']:.4f}"
            })

    # Create comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE - ALL METHODS")
    print(f"{'='*60}")

    df = pd.DataFrame(all_results)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    # Calculate average F1 for each method
    print("\n\nAverage F1 Scores by Method:")
    print("-" * 30)
    avg_f1 = df.groupby('Method')['F1'].apply(lambda x: x.astype(float).mean())
    for method, f1 in avg_f1.items():
        print(f"{method}: {f1:.4f}")

    return all_results


def main():
    """Main function - run both test and evaluation"""
    # Run single example test
    test_single_example()

    # Run full evaluation
    results = evaluate_all_datasets()

    return results


if __name__ == "__main__":
    main()