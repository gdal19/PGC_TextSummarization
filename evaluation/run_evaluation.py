from __future__ import annotations

import os
import sys
import pandas as pd

# Add the project root to sys.path to ensure absolute imports resolve correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_all import load_all_datasets
from evaluation.metrics import compute_bertscore, compute_bleu, compute_rouge
from models.run_bart import run_bart
from models.run_pegasus import run_pegasus
from models.run_t5 import run_t5
from models.run_lexrank import run_lexrank
from models.run_textrank import run_textrank


def run_evaluation() -> None:
    """Runs all summarization techniques against all datasets and evaluates them.

    Saves the tabulated results to results/results.csv.
    """
    # 1. Load all datasets
    datasets = load_all_datasets()

    # Define model runners dictionary
    runners = {
        "BART": run_bart,
        "Pegasus": run_pegasus,
        "T5": run_t5,
        "LexRank": run_lexrank,
        "TextRank": run_textrank,
    }

    results_list = []

    # 2. Iterate through each dataset and technique combination
    for dataset_name, dataset_samples in datasets.items():
        print(
            f"\n========================================\n"
            f"Evaluating on Dataset: {dataset_name.upper()}\n"
            f"========================================"
        )
        # References are the same for all models run on this dataset
        references = [sample["reference"] for sample in dataset_samples]

        for model_name, runner_fn in runners.items():
            print(f"Running technique '{model_name}' on '{dataset_name}'...")

            # Generate predictions using the technique
            predictions = runner_fn(dataset_samples)

            # Compute metrics
            print(f"Computing metrics for '{model_name}' on '{dataset_name}'...")
            rouge_scores = compute_rouge(predictions, references)
            bert_scores = compute_bertscore(predictions, references)
            bleu_scores = compute_bleu(predictions, references)

            # Tabulate results
            result_row = {
                "technique": model_name,
                "dataset": dataset_name,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "bertscore_f1": bert_scores["bertscore_f1"],
                "bleu": bleu_scores["bleu"],
            }
            results_list.append(result_row)

    # 3. Create DataFrame and export to CSV
    df = pd.DataFrame(results_list)

    # Ensure results folder exists
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nEvaluation finished. Results saved to: {csv_path}")


if __name__ == "__main__":
    run_evaluation()
