from __future__ import annotations

import os
import pandas as pd


def load_and_compare() -> None:
    """Loads results/results.csv and prints a formatted comparison table

    for each dataset, highlighting the best technique per metric.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "results.csv",
    )

    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at '{csv_path}'.")
        print("Please run 'python3 evaluation/run_evaluation.py' first.")
        return

    # Load results
    df = pd.read_csv(csv_path)

    # Metrics to compare
    metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "bleu"]

    print("\n" + "=" * 80)
    print("        TEXT SUMMARIZATION TECHNIQUES COMPARISON RESULTS")
    print("=" * 80)
    print("Note: '*' indicates the best performing technique for that metric.")
    print("=" * 80)

    # Group by dataset
    grouped = df.groupby("dataset")

    for dataset_name, group in grouped:
        print(f"\nDataset: {dataset_name.upper()}")
        print("-" * 80)

        # Find maximum values for highlighting
        max_values = {metric: group[metric].max() for metric in metrics}

        # Header
        header = f"{'Technique':<15} | {'ROUGE-1':<11} | {'ROUGE-2':<11} | {'ROUGE-L':<11} | {'BERTScore':<11} | {'BLEU':<10}"
        print(header)
        print("-" * 80)

        # Print each row with formatting and highlight if it matches the maximum
        for _, row in group.iterrows():
            tech = row["technique"]

            formatted_cols = []
            for metric in metrics:
                val = row[metric]
                # Format to 4 decimal places
                val_str = f"{val:.4f}"

                # Highlight if it's the best value (within floating point precision tolerance)
                if abs(val - max_values[metric]) < 1e-9:
                    val_str = f"{val_str}*"

                formatted_cols.append(val_str)

            row_str = f"{tech:<15} | {formatted_cols[0]:<11} | {formatted_cols[1]:<11} | {formatted_cols[2]:<11} | {formatted_cols[3]:<11} | {formatted_cols[4]:<10}"
            print(row_str)

        print("-" * 80)


if __name__ == "__main__":
    load_and_compare()
