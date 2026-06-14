from __future__ import annotations

import tqdm
from datasets import load_dataset


def load_bigpatent_dataset() -> list[dict]:
    """Loads the big_patent (section a) test split and selects the first 100 samples.

    Maps 'description' to 'source' and 'abstract' to 'reference'.
    """
    print("Loading big_patent (section a) dataset...")
    # Load the test split of big_patent for section 'a' (Human Necessities)
    dataset = load_dataset("big_patent", "a", split="test")

    # Select the first 100 samples as specified
    subset = dataset.select(range(100))

    processed_data = []
    # Use tqdm to show progress during formatting
    for sample in tqdm.tqdm(subset, desc="Processing BigPatent samples"):
        processed_data.append(
            {
                "source": sample["description"],
                "reference": sample["abstract"],
            }
        )

    return processed_data


if __name__ == "__main__":
    data = load_bigpatent_dataset()
    print(f"Successfully loaded {len(data)} BigPatent samples.")
    if data:
        print(f"Sample source snippet: {data[0]['source'][:100]}...")
        print(f"Sample reference snippet: {data[0]['reference'][:100]}...")
