from __future__ import annotations

import os
import sys
import nltk
import tqdm
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer


def ensure_nltk_resources():
    """Dynamically downloads the NLTK 'punkt' resource if not present,

    as it is required by the sumy Tokenizer.
    """
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            print("Downloading NLTK 'punkt' tokenizer resource...")
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)


def run_lexrank(data: list[dict], sentences_count: int = 3) -> list[str]:
    """Generates extractive summaries using LexRank.

    Args:
        data: A list of dicts with 'source' and 'reference' keys.
        sentences_count: Number of sentences to extract for the summary.

    Returns:
        A list of generated summary strings.
    """
    print(
        f"Running LexRank model (extracting {sentences_count} sentences)..."
    )
    ensure_nltk_resources()

    summarizer = LexRankSummarizer()
    summaries = []

    for item in tqdm.tqdm(data, desc="Running LexRank"):
        text = item["source"]

        # Parse text into documents and sentences
        parser = PlaintextParser.from_string(text, Tokenizer("english"))

        # LexRank selects the most central/important sentences
        summary_sentences = summarizer(parser.document, sentences_count)

        # Reconstruct the summary by joining the selected sentences
        summary = " ".join([str(sentence) for sentence in summary_sentences])
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
    results = run_lexrank(test_data)

    for i, (orig, summ) in enumerate(zip(test_data, results)):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Source: {orig['source'][:120]}...")
        print(f"Reference: {orig['reference'][:120]}...")
        print(f"Generated Summary: {summ}")
