from __future__ import annotations

try:
    from data.load_bigpatent import load_bigpatent_dataset
    from data.load_cnn import load_cnn_dataset
    from data.load_pubmed import load_pubmed_dataset
except ImportError:
    from load_bigpatent import load_bigpatent_dataset
    from load_cnn import load_cnn_dataset
    from load_pubmed import load_pubmed_dataset


def load_all_datasets() -> dict[str, list[dict]]:
    """Loads PubMed, BigPatent, and CNN/DailyMail datasets.

    Returns:
        dict: A dictionary containing the loaded datasets with keys
              'pubmed', 'bigpatent', and 'cnn'.
    """
    print("Starting loading process for all datasets...")
    pubmed_data = load_pubmed_dataset()
    bigpatent_data = load_bigpatent_dataset()
    cnn_data = load_cnn_dataset()

    return {
        "pubmed": pubmed_data,
        "bigpatent": bigpatent_data,
        "cnn": cnn_data,
    }


if __name__ == "__main__":
    datasets = load_all_datasets()
    for name, data in datasets.items():
        print(f"Dataset '{name}' loaded with {len(data)} samples.")
