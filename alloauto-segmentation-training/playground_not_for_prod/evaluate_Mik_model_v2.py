"""
Evaluate Binary Auto/Allo Classifier on Code-Switching Test Data
Converts sentence-level binary predictions to token-level 4-class format
Then evaluates with the same metrics as the fine-tuned CS model
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
import re
from pathlib import Path


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_binary_model_onnx(model_path):
    """Load the ONNX binary classifier"""
    print(f"Loading ONNX classifier from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load ONNX model
    onnx_path = f"{model_path}/onnx/model.onnx"
    session = ort.InferenceSession(onnx_path)

    # Print input/output info for debugging
    print(f"ONNX inputs: {[i.name for i in session.get_inputs()]}")
    print(f"ONNX outputs: {[o.name for o in session.get_outputs()]}")

    return tokenizer, session


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def split_segment_to_sentences(segment_text):
    """
    Split segment into sentences using / as delimiter (matching your data format)
    """
    # Split by / or // while keeping the delimiter
    sentences = re.split(r'(\s*/+\s*)', segment_text)

    # Reconstruct sentences with their delimiters
    result = []
    current = ""
    for part in sentences:
        if re.match(r'\s*/+\s*', part):  # This is a delimiter
            current += part
            if current.strip():
                result.append(current.strip())
            current = ""
        else:
            current += part

    if current.strip():
        result.append(current.strip())

    return result


# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_sentence_onnx(text, tokenizer, session, max_length=512):
    """Classify a sentence using the ONNX model"""
    if not text.strip():
        return 'auto'  # Default for empty

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='np',  # NumPy arrays for ONNX
        truncation=True,
        max_length=max_length,
        padding=True
    )

    # Prepare ONNX inputs - ONLY include what the model expects
    ort_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }

    # DON'T add token_type_ids - the model doesn't expect it
    # Remove these lines:
    # if 'token_type_ids' in inputs:
    #     ort_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)

    # Run inference
    try:
        outputs = session.run(None, ort_inputs)
        logits = outputs[0][0]  # First output, first batch
    except Exception as e:
        print(f"Error during inference: {e}")
        print(f"Input shapes: {[(k, v.shape) for k, v in ort_inputs.items()]}")
        raise

    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Assuming model outputs [allo, auto] based on your JS code
    allo_prob = probs[0]
    auto_prob = probs[1] if len(probs) > 1 else 1 - probs[0]

    return 'auto' if auto_prob > allo_prob else 'allo'


def convert_to_token_labels(segment_tokens, tokenizer, session):
    """
    Convert binary sentence predictions to 4-class token labels

    Classes:
    0: Non-switch Auto
    1: Non-switch Allo
    2: Switch TO Auto
    3: Switch TO Allo
    """
    # Join tokens to recreate segment text
    segment_text = ' '.join(segment_tokens)

    # Split into sentences
    sentences = split_segment_to_sentences(segment_text)

    if not sentences:
        # Default to all auto if no sentences found
        return [0] * len(segment_tokens)

    # Classify each sentence
    sentence_predictions = []
    for sent in sentences:
        pred = classify_sentence_onnx(sent, tokenizer, session)
        print("pred: ", pred)
        sentence_predictions.append(pred)

    # Convert to token-level labels
    token_labels = []
    current_mode = None
    token_idx = 0

    for sent_idx, (sent, pred) in enumerate(zip(sentences, sentence_predictions)):
        # Get tokens in this sentence
        sent_tokens = sent.split()

        for word_idx, word in enumerate(sent_tokens):
            if token_idx >= len(segment_tokens):
                break

            # For first sentence or when mode doesn't change
            if sent_idx == 0 or current_mode == pred:
                # Just assign the mode (no switch)
                if pred == 'auto':
                    token_labels.append(0)  # Non-switch Auto
                else:
                    token_labels.append(1)  # Non-switch Allo
            else:
                # Mode changed - mark switch
                if word_idx == 0:  # First word of sentence gets switch label
                    if pred == 'auto':
                        token_labels.append(2)  # Switch TO Auto
                    else:
                        token_labels.append(3)  # Switch TO Allo
                else:
                    # Rest of words get continuation
                    if pred == 'auto':
                        token_labels.append(0)  # Non-switch Auto
                    else:
                        token_labels.append(1)  # Non-switch Allo

            token_idx += 1

        current_mode = pred

    # Pad with current mode if needed
    while len(token_labels) < len(segment_tokens):
        token_labels.append(0 if current_mode == 'auto' else 1)

    return token_labels[:len(segment_tokens)]


# ============================================================================
# EVALUATION METRICS (from your original code)
# ============================================================================

def evaluate_switch_detection_with_proximity(true_labels, pred_labels, tolerance=5):
    """
    Evaluate switch detection with proximity tolerance and TYPE matching
    Switch types must match: class 2 with class 2, class 3 with class 3
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Find switch positions BY TYPE
    true_switches_to_auto = np.where(true_labels == 2)[0]  # Switch to Auto
    true_switches_to_allo = np.where(true_labels == 3)[0]  # Switch to Allo
    pred_switches_to_auto = np.where(pred_labels == 2)[0]  # Predicted Switch to Auto
    pred_switches_to_allo = np.where(pred_labels == 3)[0]  # Predicted Switch to Allo

    # Track matches separately by type
    matched_true_to_auto = set()
    matched_pred_to_auto = set()
    matched_true_to_allo = set()
    matched_pred_to_allo = set()

    exact_matches = 0
    proximity_matches = 0

    # Match "Switch to Auto" predictions with "Switch to Auto" ground truth
    for pred_pos in pred_switches_to_auto:
        if len(true_switches_to_auto) > 0:
            distances = np.abs(true_switches_to_auto - pred_pos)
            min_distance = np.min(distances)
            closest_true_idx = np.argmin(distances)
            closest_true_pos = true_switches_to_auto[closest_true_idx]

            # Only count if not already matched
            if closest_true_pos not in matched_true_to_auto:
                if min_distance == 0:
                    exact_matches += 1
                    matched_true_to_auto.add(closest_true_pos)
                    matched_pred_to_auto.add(pred_pos)
                elif min_distance <= tolerance:
                    proximity_matches += 1
                    matched_true_to_auto.add(closest_true_pos)
                    matched_pred_to_auto.add(pred_pos)

    # Match "Switch to Allo" predictions with "Switch to Allo" ground truth
    for pred_pos in pred_switches_to_allo:
        if len(true_switches_to_allo) > 0:
            distances = np.abs(true_switches_to_allo - pred_pos)
            min_distance = np.min(distances)
            closest_true_idx = np.argmin(distances)
            closest_true_pos = true_switches_to_allo[closest_true_idx]

            # Only count if not already matched
            if closest_true_pos not in matched_true_to_allo:
                if min_distance == 0:
                    exact_matches += 1
                    matched_true_to_allo.add(closest_true_pos)
                    matched_pred_to_allo.add(pred_pos)
                elif min_distance <= tolerance:
                    proximity_matches += 1
                    matched_true_to_allo.add(closest_true_pos)
                    matched_pred_to_allo.add(pred_pos)

    # Total counts
    total_true_switches = len(true_switches_to_auto) + len(true_switches_to_allo)
    total_pred_switches = len(pred_switches_to_auto) + len(pred_switches_to_allo)
    total_matched_true = len(matched_true_to_auto) + len(matched_true_to_allo)
    total_matched_pred = len(matched_pred_to_auto) + len(matched_pred_to_allo)

    total_matches = exact_matches + proximity_matches
    missed_switches = total_true_switches - total_matched_true
    false_switches = total_pred_switches - total_matched_pred

    # Calculate metrics
    precision = total_matches / total_pred_switches if total_pred_switches > 0 else 0
    recall = total_matches / total_true_switches if total_true_switches > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-type metrics
    to_auto_precision = len(matched_pred_to_auto) / len(pred_switches_to_auto) if len(pred_switches_to_auto) > 0 else 0
    to_auto_recall = len(matched_true_to_auto) / len(true_switches_to_auto) if len(true_switches_to_auto) > 0 else 0
    to_allo_precision = len(matched_pred_to_allo) / len(pred_switches_to_allo) if len(pred_switches_to_allo) > 0 else 0
    to_allo_recall = len(matched_true_to_allo) / len(true_switches_to_allo) if len(true_switches_to_allo) > 0 else 0

    return {
        'exact_matches': exact_matches,
        'proximity_matches': proximity_matches,
        'total_matches': total_matches,
        'true_switches': total_true_switches,
        'pred_switches': total_pred_switches,
        'missed_switches': missed_switches,
        'false_switches': false_switches,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # Per-type metrics
        'true_to_auto': len(true_switches_to_auto),
        'true_to_allo': len(true_switches_to_allo),
        'pred_to_auto': len(pred_switches_to_auto),
        'pred_to_allo': len(pred_switches_to_allo),
        'matched_to_auto': len(matched_true_to_auto),
        'matched_to_allo': len(matched_true_to_allo),
        'to_auto_precision': to_auto_precision,
        'to_auto_recall': to_auto_recall,
        'to_allo_precision': to_allo_precision,
        'to_allo_recall': to_allo_recall
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_binary_as_token_classifier(test_csv='test_segments.csv',
                                        model_path='./alloauto/web/model',
                                        tolerance=5):
    """
    Main evaluation - converts binary to token-level and evaluates like CS model
    """
    print("=" * 80)
    print("EVALUATING BINARY CLASSIFIER AS TOKEN-LEVEL CODE-SWITCH DETECTOR")
    print("=" * 80)

    # Load ONNX model
    tokenizer, session = load_binary_model_onnx(model_path)


    # Load test data
    print(f"\nLoading test data from {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"Found {len(test_df)} test segments")


    # Collect all predictions and labels
    all_true_labels = []
    all_pred_labels = []

    print("\nConverting binary predictions to token-level labels...")
    for idx, row in test_df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing segment {idx + 1}/{len(test_df)}")

        # Get true labels
        true_labels = [int(l) for l in row['labels'].split(',')]

        # Get tokens
        tokens = row['tokens'].split()

        # Convert binary predictions to token labels
        pred_labels = convert_to_token_labels(tokens, tokenizer, session)

        # Ensure same length
        min_len = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_len]
        pred_labels = pred_labels[:min_len]

        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)

    # Calculate overall accuracy
    all_true_array = np.array(all_true_labels)
    all_pred_array = np.array(all_pred_labels)
    accuracy = (all_true_array == all_pred_array).mean()

    # Calculate switch metrics with proximity tolerance
    switch_metrics = evaluate_switch_detection_with_proximity(
        all_true_labels, all_pred_labels, tolerance=tolerance
    )

    # Print results in same format as your CS model
    print("\n=== BINARY MODEL AS TOKEN CLASSIFIER - TEST RESULTS ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Switch F1: {switch_metrics['f1']:.3f}")
    print(f"Switch Precision: {switch_metrics['precision']:.3f}")
    print(f"Switch Recall: {switch_metrics['recall']:.3f}")
    print(f"Exact Matches: {switch_metrics['exact_matches']}")
    print(f"Proximity Matches: {switch_metrics['proximity_matches']}")
    print(f"True Switches: {switch_metrics['true_switches']}")
    print(f"Predicted Switches: {switch_metrics['pred_switches']}")

    print(f"\nPer-Type Performance:")
    print(f"  Switch→Auto Precision: {switch_metrics['to_auto_precision']:.3f}")
    print(f"  Switch→Auto Recall: {switch_metrics['to_auto_recall']:.3f}")
    print(f"  Switch→Allo Precision: {switch_metrics['to_allo_precision']:.3f}")
    print(f"  Switch→Allo Recall: {switch_metrics['to_allo_recall']:.3f}")
    print(f"  True Switch→Auto: {switch_metrics['true_to_auto']}")
    print(f"  True Switch→Allo: {switch_metrics['true_to_allo']}")
    print(f"  Matched Switch→Auto: {switch_metrics['matched_to_auto']}")
    print(f"  Matched Switch→Allo: {switch_metrics['matched_to_allo']}")

    # Compare with your CS model results
    print("\n=== COMPARISON WITH YOUR FINE-TUNED CODE-SWITCHING MODEL ===")
    print("Your CS Model Results:")
    print("  Accuracy: 0.829")
    print("  Switch F1: 0.407, Precision: 0.400, Recall: 0.413")
    print("  Switch→Auto: Precision: 0.457, Recall: 0.843")
    print("  Switch→Allo: Precision: 0.115, Recall: 0.037")

    print("\nBinary Model as Token Classifier:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(
        f"  Switch F1: {switch_metrics['f1']:.3f}, Precision: {switch_metrics['precision']:.3f}, Recall: {switch_metrics['recall']:.3f}")
    print(
        f"  Switch→Auto: Precision: {switch_metrics['to_auto_precision']:.3f}, Recall: {switch_metrics['to_auto_recall']:.3f}")
    print(
        f"  Switch→Allo: Precision: {switch_metrics['to_allo_precision']:.3f}, Recall: {switch_metrics['to_allo_recall']:.3f}")
    print_simple_comparison(test_df, tokenizer, session, num_segments=3)

    return switch_metrics


def print_simple_comparison(test_df, tokenizer, session, num_segments=3, tokens_per_segment=200):
    """
    Simple comparison showing first N tokens of each segment
    """
    print("\n" + "=" * 80)
    print("PREDICTION COMPARISON (First {} tokens per segment)".format(tokens_per_segment))
    print("=" * 80)

    for idx in range(min(num_segments, len(test_df))):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()[:tokens_per_segment]
        true_labels = [int(l) for l in row['labels'].split(',')][:tokens_per_segment]
        pred_labels = convert_to_token_labels(row['tokens'].split(), tokenizer, session)[:tokens_per_segment]

        print(f"\nSegment {idx + 1}:")
        print("Token        | True | Pred | Match")
        print("-" * 35)

        for t, tl, pl in zip(tokens, true_labels, pred_labels):
            match = "✓" if tl == pl else "✗"
            print(f"{t:12s} | {tl:4d} | {pl:4d} | {match}")



# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run evaluation with your paths
    import json

    with open("./alloauto/web/model/config.json", 'r') as f:
        config = json.load(f)
        if 'id2label' in config:
            print("Label mapping:", config['id2label'])
        if 'label2id' in config:
            print("Label to ID:", config['label2id'])
        else:
            print("Label map not found")
    results = evaluate_binary_as_token_classifier(
        test_csv='./test_segments.csv',  # Adjust path as needed
        model_path='./alloauto/web/model',
        tolerance=5  # Same 5-token tolerance as your model
    )

    print("\n" + "=" * 80)

    print("Evaluation complete!")
