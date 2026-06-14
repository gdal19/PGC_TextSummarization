from __future__ import annotations

import tqdm
from datasets import load_dataset


def load_pubmed_dataset() -> list[dict]:
    """Loads the ccdv/pubmed-summarization test split and selects the first 100 samples.

    Maps 'article' to 'source' and 'abstract' to 'reference'.
    """
    print("Loading ccdv/pubmed-summarization dataset...")
    # Load the test split of pubmed dataset
    dataset = load_dataset("ccdv/pubmed-summarization", split="test")

    # Select the first 100 samples as specified
    subset = dataset.select(range(100))

    processed_data = []
    # Use tqdm to show progress during formatting
    for sample in tqdm.tqdm(subset, desc="Processing PubMed samples"):
        processed_data.append(
            {
                "source": sample["article"],
                "reference": sample["abstract"],
            }
        )

    return processed_data


if __name__ == "__main__":
    data = load_pubmed_dataset()
    print(f"Successfully loaded {len(data)} PubMed samples.")
    if data:
        print(f"Sample source snippet: {data[0]['source'][:100]}...")
        print(f"Sample reference snippet: {data[0]['reference'][:100]}...")
