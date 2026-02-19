"""
Switch Detection Evaluation Module
Evaluates code-switching detection by comparing switch points rather than token-level accuracy.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
import json


def convert_4class_to_binary(labels):
    """
    Convert 4-class labels to binary auto/allo
    0: non_switch_auto -> 0 (auto)
    1: non_switch_allo -> 1 (allo)
    2: switch_to_auto -> 0 (auto)
    3: switch_to_allo -> 1 (allo)
    """
    binary = []
    for label in labels:
        if isinstance(label, (list, np.ndarray)):
            binary.extend([0 if l in [0, 2] else 1 for l in label])
        else:
            binary.append(0 if label in [0, 2] else 1)
    return binary


def find_change_points(arr):
    """Find positions where the label changes."""
    change_points = []
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            change_points.append(i)
    return change_points


def evaluate_switch_detection(true_labels, pred_labels, tolerance=5):
    """
    Evaluate switch detection performance with tolerance.

    Args:
        true_labels: List of true labels (4-class system)
        pred_labels: List of predicted labels (4-class system)
        tolerance: Number of tokens tolerance for matching switches

    Returns:
        Dictionary with metrics
    """
    # Convert to binary
    true_binary = convert_4class_to_binary(true_labels)
    pred_binary = convert_4class_to_binary(pred_labels)

    # Find switch points
    true_switches = find_change_points(true_binary)
    pred_switches = find_change_points(pred_binary)

    # Calculate matches with different tolerance levels
    tolerance_metrics = {}
    for tol in range(0, tolerance + 1):
        tp = 0
        matched_true = set()
        matched_pred = set()

        for p_idx, p in enumerate(pred_switches):
            for t_idx, t in enumerate(true_switches):
                if t_idx not in matched_true and abs(p - t) <= tol:
                    tp += 1
                    matched_true.add(t_idx)
                    matched_pred.add(p_idx)
                    break

        fp = len(pred_switches) - len(matched_pred)
        fn = len(true_switches) - len(matched_true)

        precision = tp / len(pred_switches) if pred_switches else 0
        recall = tp / len(true_switches) if true_switches else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        tolerance_metrics[tol] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Main metrics at specified tolerance
    main_metrics = tolerance_metrics[tolerance]

    return {
        'true_switches': true_switches,
        'pred_switches': pred_switches,
        'num_true_switches': len(true_switches),
        'num_pred_switches': len(pred_switches),
        'tolerance': tolerance,
        'tp': main_metrics['tp'],
        'fp': main_metrics['fp'],
        'fn': main_metrics['fn'],
        'precision': main_metrics['precision'],
        'recall': main_metrics['recall'],
        'f1': main_metrics['f1'],
        'tolerance_breakdown': tolerance_metrics
    }


def print_switch_evaluation(metrics):
    """Pretty print the evaluation metrics."""
    print("\n" + "=" * 60)
    print("SWITCH DETECTION EVALUATION (Binary Mode Changes)")
    print("=" * 60)

    print(f"\nSwitch Points:")
    print(
        f"  True switches: {metrics['num_true_switches']} at positions {metrics['true_switches'][:10]}{'...' if len(metrics['true_switches']) > 10 else ''}")
    print(
        f"  Predicted switches: {metrics['num_pred_switches']} at positions {metrics['pred_switches'][:10]}{'...' if len(metrics['pred_switches']) > 10 else ''}")

    print(f"\n--- PERFORMANCE AT TOLERANCE ±{metrics['tolerance']} tokens ---")
    print(f"True Positives (TP): {metrics['tp']}")
    print(f"False Positives (FP): {metrics['fp']}")
    print(f"False Negatives (FN): {metrics['fn']}")

    print(f"\nPrecision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")

    print(f"\n--- PERFORMANCE BY TOLERANCE ---")
    print("Tolerance | TP  | FP  | FN  | Precision | Recall | F1")
    print("-" * 60)
    for tol, tol_metrics in metrics['tolerance_breakdown'].items():
        print(f"    ±{tol:2d}   | {tol_metrics['tp']:3d} | {tol_metrics['fp']:3d} | {tol_metrics['fn']:3d} | "
              f"  {tol_metrics['precision']:.3f}   | {tol_metrics['recall']:.3f} | {tol_metrics['f1']:.3f}")


def evaluate_model_on_test_switch_detection(model, tokenizer, test_csv='test_sequences_combined.csv', tolerance=5):
    """
    Evaluate model focusing on switch detection rather than token classification.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_csv: Path to test CSV
        tolerance: Tolerance in tokens for switch matching

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING SWITCH DETECTION ON TEST SET")
    print("=" * 60)

    test_df = pd.read_csv(test_csv)
    device = next(model.parameters()).device
    model.eval()

    all_true_labels = []
    all_pred_labels = []
    sequence_results = []

    for idx, row in test_df.iterrows():
        # Process tokens
        raw_tokens = row['tokens'].split()
        tokens = [t for t in raw_tokens if t not in ['I-ALLO', 'B-ALLO', 'I-AUTO', 'B-AUTO', 'O']]

        true_labels = list(map(int, row['labels'].split(',')))[:len(tokens)]

        if len(tokens) == 0:
            continue

        # Get predictions
        tokenizer_output = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in tokenizer_output.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Align predictions
        word_ids = tokenizer_output.word_ids()
        aligned_predictions = []
        previous_word_idx = None

        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                aligned_predictions.append(predictions[0][i].item())
            previous_word_idx = word_idx

        aligned_predictions = aligned_predictions[:len(tokens)]

        # Convert to binary and find switches for this sequence
        seq_true_binary = convert_4class_to_binary(true_labels)
        seq_pred_binary = convert_4class_to_binary(aligned_predictions)

        seq_true_switches = find_change_points(seq_true_binary)
        seq_pred_switches = find_change_points(seq_pred_binary)

        sequence_results.append({
            'sequence_idx': idx,
            'true_switches': seq_true_switches,
            'pred_switches': seq_pred_switches,
            'num_tokens': len(tokens)
        })

        all_true_labels.extend(true_labels)
        all_pred_labels.extend(aligned_predictions)

    # Overall evaluation
    overall_metrics = evaluate_switch_detection(all_true_labels, all_pred_labels, tolerance=tolerance)
    print_switch_evaluation(overall_metrics)

    # Sequence-level statistics
    sequences_with_true_switches = sum(1 for r in sequence_results if r['true_switches'])
    sequences_with_pred_switches = sum(1 for r in sequence_results if r['pred_switches'])

    print(f"\n--- SEQUENCE-LEVEL STATISTICS ---")
    print(f"Total sequences: {len(sequence_results)}")
    print(f"Sequences with true switches: {sequences_with_true_switches}")
    print(f"Sequences with predicted switches: {sequences_with_pred_switches}")

    # Save detailed results
    results_to_save = {
        'overall_metrics': overall_metrics,
        'sequence_level_stats': {
            'total_sequences': len(sequence_results),
            'sequences_with_true_switches': sequences_with_true_switches,
            'sequences_with_pred_switches': sequences_with_pred_switches
        }
    }

    with open('switch_detection_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    print(f"\nDetailed results saved to switch_detection_results.json")

    return overall_metrics, sequence_results