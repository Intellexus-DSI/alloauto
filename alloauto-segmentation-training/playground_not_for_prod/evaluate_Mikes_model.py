"""
Convert Binary Auto/Allo Classifier predictions to 4-class token-level format
Then evaluate exactly like the code-switching model
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from pathlib import Path

"""
Convert Binary Auto/Allo Classifier predictions to 4-class token-level format
Using ONNX model instead of PyTorch
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
import re
from pathlib import Path


def load_binary_model_onnx(model_path):
    """Load the ONNX binary classifier"""
    print(f"Loading ONNX classifier from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load ONNX model
    onnx_path = f"{model_path}/onnx/model.onnx"
    session = ort.InferenceSession(onnx_path)

    # Print input/output names for debugging
    print(f"ONNX inputs: {[i.name for i in session.get_inputs()]}")
    print(f"ONNX outputs: {[o.name for o in session.get_outputs()]}")

    return tokenizer, session


def split_segment_to_sentences(segment_text):
    """Split segment into sentences using / as delimiter"""
    sentences = re.split(r'(\s*/+\s*)', segment_text)

    result = []
    current = ""
    for part in sentences:
        if re.match(r'\s*/+\s*', part):
            current += part
            if current.strip():
                result.append(current.strip())
            current = ""
        else:
            current += part

    if current.strip():
        result.append(current.strip())

    return result


def classify_sentence_onnx(text, tokenizer, session, max_length=512):
    """Classify a sentence using the ONNX model"""
    if not text.strip():
        return 'auto'

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='np',  # NumPy arrays for ONNX
        truncation=True,
        max_length=max_length,
        padding=True
    )

    # Prepare ONNX inputs - check what the model expects
    ort_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }

    # If model expects token_type_ids
    if 'token_type_ids' in inputs:
        ort_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)

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
    """Convert binary sentence predictions to 4-class token labels"""
    segment_text = ' '.join(segment_tokens)
    sentences = split_segment_to_sentences(segment_text)

    if not sentences:
        return [0] * len(segment_tokens)

    # Classify each sentence
    sentence_predictions = []
    for sent in sentences:
        pred = classify_sentence_onnx(sent, tokenizer, session)
        sentence_predictions.append(pred)

    # Convert to token-level labels
    token_labels = []
    current_mode = None
    token_idx = 0

    for sent_idx, (sent, pred) in enumerate(zip(sentences, sentence_predictions)):
        sent_tokens = sent.split()

        for word_idx, word in enumerate(sent_tokens):
            if token_idx >= len(segment_tokens):
                break

            # For first sentence or when mode doesn't change
            if sent_idx == 0 or current_mode == pred:
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


# [Keep all your evaluation functions the same - evaluate_switch_detection_with_proximity, etc.]

def evaluate_binary_as_token_classifier(test_csv='test_segments.csv', model_path='./alloauto/web/model', tolerance=5):
    """Main evaluation - converts binary to token-level and evaluates like CS model"""
    print("=" * 80)
    print("EVALUATING BINARY CLASSIFIER AS TOKEN-LEVEL CODE-SWITCH DETECTOR")
    print("=" * 80)

    # Load ONNX model
    tokenizer, session = load_binary_model_onnx(model_path)

    # Load test data
    print(f"\nLoading test data from {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"Found {len(test_df)} test segments")

    # Rest of your evaluation code remains the same...
    all_true_labels = []
    all_pred_labels = []

    print("\nConverting binary predictions to token-level labels...")
    for idx, row in test_df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing segment {idx + 1}/{len(test_df)}")

        true_labels = [int(l) for l in row['labels'].split(',')]
        tokens = row['tokens'].split()

        pred_labels = convert_to_token_labels(tokens, tokenizer, session)

        min_len = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_len]
        pred_labels = pred_labels[:min_len]

        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)

    # [Rest of evaluation code stays the same...]
    all_true_array = np.array(all_true_labels)
    all_pred_array = np.array(all_pred_labels)
    accuracy = (all_true_array == all_pred_array).mean()

    # Calculate switch metrics
    switch_metrics = evaluate_switch_detection_with_proximity(
        all_true_labels, all_pred_labels, tolerance=tolerance
    )

    # Print results exactly like your model
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

    # Compare with your CS model
    print("\n=== COMPARISON WITH YOUR CODE-SWITCHING MODEL ===")
    print("Your CS Model:")
    print("  Switch F1: 0.407, Precision: 0.400, Recall: 0.413")
    print("Binary as Token Classifier:")
    print(
        f"  Switch F1: {switch_metrics['f1']:.3f}, Precision: {switch_metrics['precision']:.3f}, Recall: {switch_metrics['recall']:.3f}")

    return switch_metrics

def evaluate_binary_as_token_classifier(test_csv='test_segments.csv', model_path='./model', tolerance=5):
    """
    Main evaluation - converts binary to token-level and evaluates like CS model
    """
    print("=" * 80)
    print("EVALUATING BINARY CLASSIFIER AS TOKEN-LEVEL CODE-SWITCH DETECTOR")
    print("=" * 80)

    # Load model
    tokenizer, model = load_binary_model(model_path)

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
        pred_labels = convert_to_token_labels(tokens, tokenizer, model)

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

    # Calculate switch metrics
    switch_metrics = evaluate_switch_detection_with_proximity(
        all_true_labels, all_pred_labels, tolerance=tolerance
    )

    # Print results exactly like your model
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

    # Compare with your CS model
    print("\n=== COMPARISON WITH YOUR CODE-SWITCHING MODEL ===")
    print("Your CS Model:")
    print("  Switch F1: 0.407, Precision: 0.400, Recall: 0.413")
    print("Binary as Token Classifier:")
    print(
        f"  Switch F1: {switch_metrics['f1']:.3f}, Precision: {switch_metrics['precision']:.3f}, Recall: {switch_metrics['recall']:.3f}")

    return switch_metrics


if __name__ == "__main__":
    results = evaluate_binary_as_token_classifier(
        test_csv='test_segments.csv',
        model_path='./alloauto/web/model',  # Your binary classifier path
        tolerance=5
    )