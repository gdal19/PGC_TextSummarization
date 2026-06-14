from __future__ import annotations

import bert_score
from rouge_score import rouge_scorer
import sacrebleu


def compute_rouge(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predictions: List of generated summary strings.
        references: List of reference summary strings.

    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1 averages.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1_f1_scores = []
    r2_f1_scores = []
    rl_f1_scores = []

    for pred, ref in zip(predictions, references):
        # rouge_scorer.score takes (target/ref, prediction)
        scores = scorer.score(ref, pred)
        r1_f1_scores.append(scores["rouge1"].fmeasure)
        r2_f1_scores.append(scores["rouge2"].fmeasure)
        rl_f1_scores.append(scores["rougeL"].fmeasure)

    # Return averaged scores
    return {
        "rouge1": sum(r1_f1_scores) / len(r1_f1_scores) if r1_f1_scores else 0.0,
        "rouge2": sum(r2_f1_scores) / len(r2_f1_scores) if r2_f1_scores else 0.0,
        "rougeL": sum(rl_f1_scores) / len(rl_f1_scores) if rl_f1_scores else 0.0,
    }


def compute_bertscore(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Computes mean BERTScore F1 score.

    Args:
        predictions: List of generated summary strings.
        references: List of reference summary strings.

    Returns:
        dict: A dictionary containing the mean bertscore_f1.
    """
    # bert_score.score takes predictions, references and evaluates on the specified language
    # Force cpu execution to align with CPU-only constraints
    _, _, f1_scores = bert_score.score(
        predictions,
        references,
        lang="en",
        model_type="distilbert-base-uncased",  # Use a small and fast model for CPU evaluation
        verbose=False,
    )
    mean_f1 = f1_scores.mean().item()

    return {"bertscore_f1": mean_f1}


def compute_bleu(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Computes corpus-level BLEU score using sacrebleu.

    Args:
        predictions: List of generated summary strings.
        references: List of reference summary strings.

    Returns:
        dict: A dictionary containing the corpus-level BLEU score.
    """
    # sacrebleu.corpus_bleu expects references to be a list of lists of references
    # (one list of reference alternatives per prediction).
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    return {"bleu": bleu.score}


if __name__ == "__main__":
    # Small test validation
    test_preds = ["The cat sat on the mat.", "I love machine learning."]
    test_refs = ["The cat sat on a mat.", "I enjoy machine learning."]

    print("Running quick test metrics evaluation...")
    r_scores = compute_rouge(test_preds, test_refs)
    b_scores = compute_bertscore(test_preds, test_refs)
    bl_scores = compute_bleu(test_preds, test_refs)

    print("ROUGE scores:", r_scores)
    print("BERTScore scores:", b_scores)
    print("BLEU scores:", bl_scores)
