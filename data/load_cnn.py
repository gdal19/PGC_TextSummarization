from __future__ import annotations

import tqdm
from datasets import load_dataset


def load_cnn_dataset() -> list[dict]:
    """Loads the cnn_dailymail (3.0.0) test split and selects the first 100 samples.

    Maps 'article' to 'source' and 'highlights' to 'reference'.
    """
    print("Loading cnn_dailymail (3.0.0) dataset...")
    # Load the test split of cnn_dailymail
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    # Select the first 100 samples as specified
    subset = dataset.select(range(100))

    processed_data = []
    # Use tqdm to show progress during formatting
    for sample in tqdm.tqdm(subset, desc="Processing CNN/DailyMail samples"):
        processed_data.append(
            {
                "source": sample["article"],
                "reference": sample["highlights"],
            }
        )

    return processed_data


if __name__ == "__main__":
    data = load_cnn_dataset()
    print(f"Successfully loaded {len(data)} CNN/DailyMail samples.")
    if data:
        print(f"Sample source snippet: {data[0]['source'][:100]}...")
        print(f"Sample reference snippet: {data[0]['reference'][:100]}...")
