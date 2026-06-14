from __future__ import annotations

import os
import sys
import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def run_pegasus(data: list[dict]) -> list[str]:
    """Generates summaries using google/pegasus-xsum.

    Args:
        data: A list of dicts with 'source' and 'reference' keys.

    Returns:
        A list of generated summary strings.
    """
    model_name = "google/pegasus-xsum"
    print(f"Running {model_name} model...")

    # Load tokenizer and model, forcing CPU device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")

    summaries = []
    # Process one-by-one to avoid out-of-memory errors and handle individual texts
    for item in tqdm.tqdm(data, desc=f"Running {model_name} on CPU"):
        text = item["source"]

        # Explicitly truncate input to Pegasus's maximum position embedding size (512)
        inputs = tokenizer(
            text, max_length=512, truncation=True, return_tensors="pt"
        ).to("cpu")

        # Generate summary with explicit length limits
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=256,
            min_length=30,
            num_beams=4,
            early_stopping=True,
        )

        # Decode tokens to string
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return summaries


if __name__ == "__main__":
    # Add project root to python path to import data modules when running directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.load_cnn import load_cnn_dataset

    # Load CNN dataset and select first 5 samples for the smoke test
    raw_data = load_cnn_dataset()
    test_data = raw_data[:5]

    print(f"Running smoke test on {len(test_data)} samples...")
    results = run_pegasus(test_data)

    for i, (orig, summ) in enumerate(zip(test_data, results)):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Source: {orig['source'][:120]}...")
        print(f"Reference: {orig['reference'][:120]}...")
        print(f"Generated Summary: {summ}")
