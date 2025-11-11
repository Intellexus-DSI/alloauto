"""
Unified evaluation script with sklearn's fbeta_score and full test set comparison
"""

import numpy as np
import pandas as pd
import torch
import re
import onnxruntime as ort
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def process_binary_model_sentence_level(tokens, tokenizer, session):
    """
    Process tokens through binary model which works at sentence level
    The model classifies chunks/sentences, not individual tokens
    """
    # Reconstruct text from tokens
    text = " ".join(tokens)

    # Split into chunks/sentences (following the JS implementation)
    # The JS splits by / or //
    chunks = re.split(r'\s*/+\s*', text)
    chunks = [c.strip() for c in chunks if c.strip()]

    if not chunks:
        # If no chunks, treat entire text as one chunk
        chunks = [text]

    # Classify each chunk
    chunk_predictions = []

    for chunk in chunks:
        if not chunk:
            continue

        # Tokenize the chunk
        inputs = tokenizer(
            chunk,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=512
        )

        # Get ONNX inputs
        onnx_inputs = {inp.name for inp in session.get_inputs()}
        filtered_inputs = {}
        for key, value in inputs.items():
            if key in onnx_inputs:
                filtered_inputs[key] = value

        # Run inference
        outputs = session.run(None, filtered_inputs)
        logits = outputs[0]

        # Apply softmax to get probabilities
        if len(logits.shape) == 2:
            logits = logits[0]  # Remove batch dimension if present

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Class 0 is allo, class 1 is auto (based on the JS code)
        predicted_class = np.argmax(probs)
        chunk_predictions.append(predicted_class)

    # Now map chunk-level predictions back to token level
    token_labels = []
    token_idx = 0

    for chunk_idx, chunk in enumerate(chunks):
        chunk_tokens = chunk.split()
        chunk_class = chunk_predictions[chunk_idx] if chunk_idx < len(chunk_predictions) else 0

        # Determine if this is a switch
        is_switch = False
        if chunk_idx > 0 and chunk_idx < len(chunk_predictions):
            prev_class = chunk_predictions[chunk_idx - 1]
            curr_class = chunk_predictions[chunk_idx]
            if prev_class != curr_class:
                is_switch = True

        # Assign labels to tokens in this chunk
        for i, token in enumerate(chunk_tokens):
            if token_idx < len(tokens):
                if i == 0 and is_switch:
                    # First token of a chunk that switches
                    if chunk_class == 0:  # Switching to allo
                        label = 3  # Switch to Allo
                    else:  # Switching to auto
                        label = 2  # Switch to Auto
                else:
                    # Continuation in current mode
                    if chunk_class == 0:  # Allo
                        label = 1  # Non-switch Allo
                    else:  # Auto
                        label = 0  # Non-switch Auto

                token_labels.append(label)
                token_idx += 1

    # Handle any remaining tokens (shouldn't happen but just in case)
    while len(token_labels) < len(tokens):
        token_labels.append(0)  # Default to Auto

    return token_labels[:len(tokens)]


def evaluate_switch_detection_with_proximity(true_labels, pred_labels, tolerance=5):
    """
    Evaluate with proximity tolerance - ALL metrics use 5-token tolerance for matching
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Find switch positions BY TYPE
    true_switches_to_auto = np.where(true_labels == 2)[0]
    true_switches_to_allo = np.where(true_labels == 3)[0]
    pred_switches_to_auto = np.where(pred_labels == 2)[0]
    pred_switches_to_allo = np.where(pred_labels == 3)[0]

    # Track matches separately by type (WITH TOLERANCE)
    matched_true_to_auto = set()
    matched_pred_to_auto = set()
    matched_true_to_allo = set()
    matched_pred_to_allo = set()

    exact_matches = 0
    proximity_matches = 0

    # [Keep all the matching logic as before - Match with tolerance]
    for pred_pos in pred_switches_to_auto:
        if len(true_switches_to_auto) > 0:
            distances = np.abs(true_switches_to_auto - pred_pos)
            min_distance = np.min(distances)
            closest_true_idx = np.argmin(distances)
            closest_true_pos = true_switches_to_auto[closest_true_idx]

            if closest_true_pos not in matched_true_to_auto and min_distance <= tolerance:
                if min_distance == 0:
                    exact_matches += 1
                else:
                    proximity_matches += 1
                matched_true_to_auto.add(closest_true_pos)
                matched_pred_to_auto.add(pred_pos)

    for pred_pos in pred_switches_to_allo:
        if len(true_switches_to_allo) > 0:
            distances = np.abs(true_switches_to_allo - pred_pos)
            min_distance = np.min(distances)
            closest_true_idx = np.argmin(distances)
            closest_true_pos = true_switches_to_allo[closest_true_idx]

            if closest_true_pos not in matched_true_to_allo and min_distance <= tolerance:
                if min_distance == 0:
                    exact_matches += 1
                else:
                    proximity_matches += 1
                matched_true_to_allo.add(closest_true_pos)
                matched_pred_to_allo.add(pred_pos)

    # Total counts
    total_true_switches = len(true_switches_to_auto) + len(true_switches_to_allo)
    total_pred_switches = len(pred_switches_to_auto) + len(pred_switches_to_allo)
    total_matches = exact_matches + proximity_matches

    # Overall metrics WITH TOLERANCE
    proximity_precision = total_matches / total_pred_switches if total_pred_switches > 0 else 0
    proximity_recall = total_matches / total_true_switches if total_true_switches > 0 else 0
    proximity_f1 = 2 * proximity_precision * proximity_recall / (proximity_precision + proximity_recall) if (
                                                                                                                        proximity_precision + proximity_recall) > 0 else 0

    beta = 2
    proximity_fbeta2 = ((1 + beta ** 2) * proximity_precision * proximity_recall /
                        (beta ** 2 * proximity_precision + proximity_recall)) if (
                                                                                             proximity_precision + proximity_recall) > 0 else 0

    # Per-type metrics WITH TOLERANCE
    to_auto_prox_precision = len(matched_pred_to_auto) / len(pred_switches_to_auto) if len(
        pred_switches_to_auto) > 0 else 0
    to_auto_prox_recall = len(matched_true_to_auto) / len(true_switches_to_auto) if len(
        true_switches_to_auto) > 0 else 0
    to_auto_prox_f1 = 2 * to_auto_prox_precision * to_auto_prox_recall / (
                to_auto_prox_precision + to_auto_prox_recall) if (
                                                                             to_auto_prox_precision + to_auto_prox_recall) > 0 else 0
    to_auto_prox_fbeta2 = ((1 + beta ** 2) * to_auto_prox_precision * to_auto_prox_recall /
                           (beta ** 2 * to_auto_prox_precision + to_auto_prox_recall)) if (
                                                                                                      to_auto_prox_precision + to_auto_prox_recall) > 0 else 0

    to_allo_prox_precision = len(matched_pred_to_allo) / len(pred_switches_to_allo) if len(
        pred_switches_to_allo) > 0 else 0
    to_allo_prox_recall = len(matched_true_to_allo) / len(true_switches_to_allo) if len(
        true_switches_to_allo) > 0 else 0
    to_allo_prox_f1 = 2 * to_allo_prox_precision * to_allo_prox_recall / (
                to_allo_prox_precision + to_allo_prox_recall) if (
                                                                             to_allo_prox_precision + to_allo_prox_recall) > 0 else 0
    to_allo_prox_fbeta2 = ((1 + beta ** 2) * to_allo_prox_precision * to_allo_prox_recall /
                           (beta ** 2 * to_allo_prox_precision + to_allo_prox_recall)) if (
                                                                                                      to_allo_prox_precision + to_allo_prox_recall) > 0 else 0

    # Calculate MACRO averages
    macro_proximity_precision = (to_auto_prox_precision + to_allo_prox_precision) / 2
    macro_proximity_recall = (to_auto_prox_recall + to_allo_prox_recall) / 2
    macro_proximity_f1 = (to_auto_prox_f1 + to_allo_prox_f1) / 2
    macro_proximity_fbeta2 = (to_auto_prox_fbeta2 + to_allo_prox_fbeta2) / 2

    # Exact metrics (NO tolerance) - for backward compatibility
    true_binary = (true_labels >= 2).astype(int)
    pred_binary = (pred_labels >= 2).astype(int)

    exact_precision = precision_score(true_binary, pred_binary, zero_division=0)
    exact_recall = recall_score(true_binary, pred_binary, zero_division=0)
    exact_f1 = f1_score(true_binary, pred_binary, zero_division=0)
    exact_fbeta2 = fbeta_score(true_binary, pred_binary, beta=2, zero_division=0)

    # Calculate exact macro metrics (if needed for backward compatibility)
    switch_labels = [2, 3]
    try:
        exact_macro_f1 = f1_score(true_labels, pred_labels, labels=switch_labels, average='macro', zero_division=0)
        exact_macro_fbeta2 = fbeta_score(true_labels, pred_labels, labels=switch_labels, average='macro', beta=2,
                                         zero_division=0)
    except:
        exact_macro_f1 = 0
        exact_macro_fbeta2 = 0

    return {
        # Overall proximity metrics (WITH TOLERANCE)
        'proximity_precision': proximity_precision,
        'proximity_recall': proximity_recall,
        'proximity_f1': proximity_f1,
        'proximity_fbeta2': proximity_fbeta2,
        'proximity_macro_fbeta2': macro_proximity_fbeta2,

        # Per-class metrics WITH TOLERANCE
        'to_auto_proximity_precision': to_auto_prox_precision,
        'to_auto_proximity_recall': to_auto_prox_recall,
        'to_auto_proximity_f1': to_auto_prox_f1,
        'to_auto_proximity_fbeta2': to_auto_prox_fbeta2,

        'to_allo_proximity_precision': to_allo_prox_precision,
        'to_allo_proximity_recall': to_allo_prox_recall,
        'to_allo_proximity_f1': to_allo_prox_f1,
        'to_allo_proximity_fbeta2': to_allo_prox_fbeta2,

        # ADD THESE ALIASES FOR BACKWARD COMPATIBILITY
        'to_auto_fbeta2': to_auto_prox_fbeta2,
        'to_allo_fbeta2': to_allo_prox_fbeta2,

        # Macro metrics
        'macro_precision': macro_proximity_precision,
        'macro_recall': macro_proximity_recall,
        'macro_f1': macro_proximity_f1,
        'macro_fbeta2': macro_proximity_fbeta2,

        # Exact metrics (NO tolerance)
        'exact_precision': exact_precision,
        'exact_recall': exact_recall,
        'exact_f1': exact_f1,
        'exact_fbeta2': exact_fbeta2,
        'exact_macro_f1': exact_macro_f1,
        'exact_macro_fbeta2': exact_macro_fbeta2,

        # Counts
        'exact_matches': exact_matches,
        'proximity_matches': proximity_matches,
        'total_matches': total_matches,
        'true_switches': total_true_switches,
        'pred_switches': total_pred_switches,
        'true_to_auto': len(true_switches_to_auto),
        'true_to_allo': len(true_switches_to_allo),
        'pred_to_auto': len(pred_switches_to_auto),
        'pred_to_allo': len(pred_switches_to_allo),
        'matched_to_auto': len(matched_true_to_auto),
        'matched_to_allo': len(matched_true_to_allo),
    }


def convert_binary_to_4class(binary_preds):
    """Convert binary predictions to 4-class labels"""
    labels_4class = []
    prev_binary = None

    for curr_binary in binary_preds:
        if prev_binary is None:
            labels_4class.append(0 if curr_binary == 0 else 1)
        else:
            if prev_binary == 0 and curr_binary == 1:
                labels_4class.append(3)  # Switch to Allo
            elif prev_binary == 1 and curr_binary == 0:
                labels_4class.append(2)  # Switch to Auto
            else:
                labels_4class.append(0 if curr_binary == 0 else 1)
        prev_binary = curr_binary

    return labels_4class


def process_binary_model(tokens, tokenizer, session):
    """Process tokens through binary model and return 4-class predictions"""
    inputs = tokenizer(
        " ".join(tokens),
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=512
    )

    # Get the input names expected by the ONNX model
    onnx_inputs = {inp.name for inp in session.get_inputs()}

    # Filter inputs to only include what ONNX expects
    filtered_inputs = {}
    for key, value in inputs.items():
        if key in onnx_inputs:
            filtered_inputs[key] = value

    # Run inference with filtered inputs
    outputs = session.run(None, filtered_inputs)
    predictions = outputs[0]

    # Handle both single and batch predictions
    if len(predictions.shape) == 3:  # Batch dimension exists
        predicted_classes = np.argmax(predictions, axis=-1)[0]
    elif len(predictions.shape) == 2:  # No batch dimension
        predicted_classes = np.argmax(predictions, axis=-1)
    else:
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")

    # Ensure predicted_classes is always an array
    if predicted_classes.ndim == 0:  # It's a scalar
        predicted_classes = np.array([predicted_classes])

    word_ids = inputs.word_ids()
    aligned_preds = []
    previous_word_idx = None

    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            if i < len(predicted_classes):
                aligned_preds.append(predicted_classes[i])
            else:
                # If we run out of predictions, use the last one
                aligned_preds.append(predicted_classes[-1] if len(predicted_classes) > 0 else 0)
        previous_word_idx = word_idx

    return convert_binary_to_4class(aligned_preds)
def process_binary_model_old(tokens, tokenizer, session):
    """Process tokens through binary model and return 4-class predictions"""
    inputs = tokenizer(
        " ".join(tokens),
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=512
    )

    outputs = session.run(None, dict(inputs))
    predictions = outputs[0]
    predicted_classes = np.argmax(predictions, axis=-1)[0]

    word_ids = inputs.word_ids()
    aligned_preds = []
    previous_word_idx = None

    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            aligned_preds.append(predicted_classes[i])
        previous_word_idx = word_idx

    return convert_binary_to_4class(aligned_preds)


def process_finetuned_model(tokens, tokenizer, model, device):
    """Process tokens through fine-tuned model"""
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

    word_ids = tokenizer_output.word_ids()
    aligned_preds = []
    previous_word_idx = None

    for j, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            aligned_preds.append(predictions[0][j].item())
        previous_word_idx = word_idx

    return aligned_preds


def verify_no_tags_in_datasets():
    """
    Verify that no <auto>, <AUTO>, <allo>, <ALLO> tags exist in the datasets
    """
    print("\n" + "=" * 80)
    print("VERIFYING NO TAGS IN DATASETS")
    print("=" * 80)

    # Define tag patterns to search for
    tag_patterns = [
        r'<auto>',
        r'<AUTO>',
        r'<allo>',
        r'<ALLO>',
        r'<\s*auto\s*>',  # With spaces
        r'<\s*AUTO\s*>',
        r'<\s*allo\s*>',
        r'<\s*ALLO\s*>'
    ]

    # Check all three dataset files
    datasets = {
        'Train': './dataset/annotated-data/train_segments.csv',
        'Validation': './dataset/annotated-data/val_segments.csv',
        'Test': './dataset/annotated-data/test_segments.csv'
    }

    all_clean = True

    for dataset_name, filepath in datasets.items():
        print(f"\nChecking {dataset_name} dataset: {filepath}")

        try:
            df = pd.read_csv(filepath)

            # Check tokens column
            tokens_with_tags = []
            segments_with_tags = []

            for idx, row in df.iterrows():
                tokens = row['tokens'].split()

                # Check each token for any tag pattern
                for token_idx, token in enumerate(tokens):
                    for pattern in tag_patterns:
                        if re.search(pattern, token, re.IGNORECASE):
                            tokens_with_tags.append((idx, token_idx, token))
                            segments_with_tags.append(idx)
                            all_clean = False
                            break

            # Also check the entire tokens string
            for idx, row in df.iterrows():
                tokens_str = row['tokens']
                for pattern in tag_patterns:
                    if re.search(pattern, tokens_str, re.IGNORECASE):
                        if idx not in segments_with_tags:
                            segments_with_tags.append(idx)
                            all_clean = False

            # Report findings
            if tokens_with_tags:
                print(f"  ⚠️ FOUND TAGS in {len(set(segments_with_tags))} segments!")
                print(f"  First 5 occurrences:")
                for seg_idx, token_idx, token in tokens_with_tags[:5]:
                    print(f"    Segment {seg_idx}, Token {token_idx}: '{token}'")
            else:
                print(f"  ✅ No tags found in tokens")

            # Check if 'original_text' column exists and verify it too
            if 'original_text' in df.columns:
                texts_with_tags = []
                for idx, row in df.iterrows():
                    if pd.notna(row['original_text']):
                        for pattern in tag_patterns:
                            if re.search(pattern, str(row['original_text']), re.IGNORECASE):
                                texts_with_tags.append(idx)
                                all_clean = False
                                break

                if texts_with_tags:
                    print(f"  ⚠️ FOUND TAGS in original_text column in {len(texts_with_tags)} segments!")
                    print(f"  Segments with tags: {texts_with_tags[:5]}...")
                else:
                    print(f"  ✅ No tags found in original_text")

        except FileNotFoundError:
            print(f"  ❌ File not found: {filepath}")
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")

    print("\n" + "=" * 80)
    if all_clean:
        print("✅ VERIFICATION PASSED: No tags found in any dataset!")
    else:
        print("⚠️ VERIFICATION FAILED: Tags found in datasets!")
        print("Please re-run preprocessing to ensure tags are removed.")
    print("=" * 80)

    return all_clean

def unified_evaluation():
    """Main evaluation function on complete test set"""
    print("=" * 80)
    print("VERIFYING DATASETS ARE TAG-FREE")
    print("=" * 80)

    datasets_clean = verify_no_tags_in_datasets()
    if not datasets_clean:
        print("\n⚠️ ERROR: Tags found in datasets!")
        print("Please re-run preprocessing to remove tags.")
        return None, None
    # Configuration
    # TEST_FILE = './test_segments.csv'
    TEST_FILE = './dataset/annotated-data/test_segments.csv'
    TOLERANCE = 5

    print(f"Loading test data from: {TEST_FILE}")
    test_df = pd.read_csv(TEST_FILE)
    print(f"Test set size: {len(test_df)} segments")

    # Calculate test set statistics
    total_tokens = 0
    total_switches = 0
    for idx in range(len(test_df)):
        labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
        total_tokens += len(labels)
        total_switches += sum(1 for l in labels if l in [2, 3])

    print(f"Total tokens in test set: {total_tokens}")
    print(f"Total switches in test set: {total_switches}")
    print(f"Average switches per segment: {total_switches / len(test_df):.2f}\n")

    # Load Binary Model
    print("Loading Binary Model...")
    binary_tokenizer = AutoTokenizer.from_pretrained('./alloauto-presentation/web/model')
    binary_session = ort.InferenceSession('./alloauto-presentation/web/model/onnx/model.onnx')

    # Load Fine-tuned Model
    print("Loading Fine-tuned Model...")
    model_id = "./tibetan_code_switching_constrained_model_wylie-final_all_data_no_labels_no_prtial_v2_2_10/final_model"
    # model_id = "levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_all_data_no_labels_no_partial"
    ft_tokenizer = AutoTokenizer.from_pretrained(model_id)
    ft_model = AutoModelForTokenClassification.from_pretrained(model_id)
    ft_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_model = ft_model.to(device)
    print(f"Using device: {device}\n")

    # Process all test segments
    print("Processing all test segments...")
    all_true_labels = []
    binary_all_pred = []
    finetuned_all_pred = []

    for idx, row in test_df.iterrows():
        if idx % 20 == 0:
            print(f"  Processing segment {idx}/{len(test_df)}...")

        tokens = row['tokens'].split()
        true_labels = [int(l) for l in row['labels'].split(',')]

        # Get predictions from both models
        # Use the corrected sentence-level function for binary model
        binary_pred = process_binary_model_sentence_level(tokens, binary_tokenizer, binary_session)
        ft_pred = process_finetuned_model(tokens, ft_tokenizer, ft_model, device)

        # Align all to same length
        min_len = min(len(true_labels), len(binary_pred), len(ft_pred))

        all_true_labels.extend(true_labels[:min_len])
        binary_all_pred.extend(binary_pred[:min_len])
        finetuned_all_pred.extend(ft_pred[:min_len])

    print(f"\nTotal tokens evaluated: {len(all_true_labels)}")
    print(f"Ground truth switches: {sum(1 for l in all_true_labels if l in [2, 3])}")
    print(f"Binary predicted switches: {sum(1 for l in binary_all_pred if l in [2, 3])}")
    print(f"Fine-tuned predicted switches: {sum(1 for l in finetuned_all_pred if l in [2, 3])}")

    # Calculate metrics for both models
    print("\nCalculating metrics...")
    binary_metrics = evaluate_switch_detection_with_proximity(all_true_labels, binary_all_pred, TOLERANCE)
    finetuned_metrics = evaluate_switch_detection_with_proximity(all_true_labels, finetuned_all_pred, TOLERANCE)

    print(f"\n{'=' * 100}")
    print("COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'=' * 100}")

    print(f"\n{'─' * 100}")
    print("PROXIMITY-BASED METRICS (5-token tolerance)")
    print(f"{'─' * 100}")
    print(f"{'Metric':<30} {'Binary Model':<20} {'ALTO BeRT':<20} {'Difference':<20}")
    print("-" * 100)

    proximity_metrics = [
        ('Precision (w/ tolerance)', 'proximity_precision'),
        ('Recall (w/ tolerance)', 'proximity_recall'),
        ('F1 (w/ tolerance)', 'proximity_f1'),
    ]

    for display, key in proximity_metrics:
        b_val = binary_metrics[key]
        f_val = finetuned_metrics[key]
        diff = f_val - b_val
        print(f"{display:<30} {b_val:<20.3f} {f_val:<20.3f} {diff:+20.3f}")

    print(f"\n{'─' * 100}")
    print("EXACT METRICS (sklearn - no tolerance)")
    print(f"{'─' * 100}")
    print(f"{'Metric':<30} {'Binary Model':<20} {'ALTO BeRT':<20} {'Difference':<20}")
    print("-" * 100)
    exact_metrics = [
        ('Precision (exact)', 'exact_precision'),  # Changed from 'sklearn_precision'
        ('Recall (exact)', 'exact_recall'),  # Changed from 'sklearn_recall'
        ('F1 (exact)', 'exact_f1'),  # Changed from 'sklearn_f1'
        ('F-beta(2) (exact)', 'exact_fbeta2'),  # Changed from 'sklearn_fbeta2'
    ]

    key_metrics = ['proximity_f1', 'exact_fbeta2', 'proximity_macro_fbeta2']  # Changed from 'sklearn_fbeta2'

    for display, key in exact_metrics:
        b_val = binary_metrics[key]
        f_val = finetuned_metrics[key]
        diff = f_val - b_val
        print(f"{display:<30} {b_val:<20.3f} {f_val:<20.3f} {diff:+20.3f}")

    print(f"\n{'─' * 100}")
    print("MACRO METRICS (average of switch types)")
    print(f"{'─' * 100}")
    print(f"{'Metric':<30} {'Binary Model':<20} {'ALTO BeRT':<20} {'Difference':<20}")
    print("-" * 100)

    macro_metrics = [
        ('Macro Precision', 'macro_precision'),
        ('Macro Recall', 'macro_recall'),
        ('Macro F1', 'macro_f1'),
        ('Macro F-beta(2)', 'macro_fbeta2'),
    ]

    for display, key in macro_metrics:
        b_val = binary_metrics[key]
        f_val = finetuned_metrics[key]
        diff = f_val - b_val
        print(f"{display:<30} {b_val:<20.3f} {f_val:<20.3f} {diff:+20.3f}")

    print(f"\n{'─' * 100}")
    print("PER-TYPE F-BETA(2) SCORES")
    print(f"{'─' * 100}")
    print(f"{'Switch Type':<30} {'Binary Model':<20} {'ALTO BeRT':<20} {'Difference':<20}")
    print("-" * 100)

    print(f"{'Switch→Auto F-beta(2)':<30} {binary_metrics['to_auto_fbeta2']:<20.3f} "
          f"{finetuned_metrics['to_auto_fbeta2']:<20.3f} "
          f"{finetuned_metrics['to_auto_fbeta2'] - binary_metrics['to_auto_fbeta2']:+20.3f}")
    print(f"{'Switch→Allo F-beta(2)':<30} {binary_metrics['to_allo_fbeta2']:<20.3f} "
          f"{finetuned_metrics['to_allo_fbeta2']:<20.3f} "
          f"{finetuned_metrics['to_allo_fbeta2'] - binary_metrics['to_allo_fbeta2']:+20.3f}")

    print(f"\n{'─' * 100}")
    print("COUNT STATISTICS")
    print(f"{'─' * 100}")
    print(f"{'Statistic':<30} {'Binary Model':<20} {'ALTO BeRT':<20}")
    print("-" * 100)

    count_stats = [
        ('True Switches', 'true_switches'),
        ('Predicted Switches', 'pred_switches'),
        ('Exact Matches', 'exact_matches'),
        ('Proximity Matches', 'proximity_matches'),
        ('True Switch→Auto', 'true_to_auto'),
        ('True Switch→Allo', 'true_to_allo'),
        ('Pred Switch→Auto', 'pred_to_auto'),
        ('Pred Switch→Allo', 'pred_to_allo'),
    ]

    for display, key in count_stats:
        b_val = binary_metrics[key]
        f_val = finetuned_metrics[key]
        print(f"{display:<30} {b_val:<20} {f_val:<20}")

    # Summary winner
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")

    winner_count = {'binary': 0, 'finetuned': 0}

    # key_metrics = ['proximity_f1', 'sklearn_fbeta2', 'macro_fbeta2']
    key_metrics = ['proximity_f1', 'exact_fbeta2', 'proximity_macro_fbeta2']  # Changed from 'sklearn_fbeta2'

    for key in key_metrics:
        if finetuned_metrics[key] > binary_metrics[key]:
            winner_count['finetuned'] += 1
        else:
            winner_count['binary'] += 1

    if winner_count['finetuned'] > winner_count['binary']:
        print("🏆 Fine-tuned model performs better on key metrics (F1, F-beta scores)")
    else:
        print("🏆 Binary model performs better on key metrics (F1, F-beta scores)")

    return binary_metrics, finetuned_metrics

def show_detailed_comparisons(test_df, binary_tokenizer, binary_session, ft_tokenizer, ft_model, num_examples=5):
    """Show detailed side-by-side comparisons of model predictions"""

    device = next(ft_model.parameters()).device
    label_names = {
        0: 'Auto',
        1: 'Allo',
        2: '→AUTO',
        3: '→ALLO'
    }

    print("\n" + "=" * 100)
    print("DETAILED SEGMENT COMPARISONS")
    print("=" * 100)

    # Sample random examples
    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    for ex_num, idx in enumerate(sample_indices):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()
        true_labels = [int(l) for l in row['labels'].split(',')]

        # Get predictions from both models
        binary_pred = process_binary_model(tokens, binary_tokenizer, binary_session)
        ft_pred = process_finetuned_model(tokens, ft_tokenizer, ft_model, device)

        # Align to same length
        min_len = min(len(tokens), len(true_labels), len(binary_pred), len(ft_pred))
        tokens = tokens[:min_len]
        true_labels = true_labels[:min_len]
        binary_pred = binary_pred[:min_len]
        ft_pred = ft_pred[:min_len]

        print(f"\n{'─' * 100}")
        print(f"EXAMPLE {ex_num + 1} - File: {row['source_file'][:50]}...")
        print(f"Segment length: {len(tokens)} tokens")

        # Count switches
        true_switches = sum(1 for l in true_labels if l in [2, 3])
        binary_switches = sum(1 for l in binary_pred if l in [2, 3])
        ft_switches = sum(1 for l in ft_pred if l in [2, 3])

        print(f"\nSwitch counts:")
        print(f"  Ground truth: {true_switches} switches")
        print(f"  Binary model: {binary_switches} switches")
        print(f"  Fine-tuned:   {ft_switches} switches")

        # Show first 30 tokens in detail
        print(f"\nDetailed token-by-token comparison (first 30 tokens):")
        print(f"{'Pos':<5} {'Token':<20} {'True':<8} {'Binary':<8} {'Fine-tuned':<12} {'Match'}")
        print("─" * 70)

        for i in range(min(30, len(tokens))):
            token = tokens[i][:19]
            true_lab = label_names[true_labels[i]]
            binary_lab = label_names[binary_pred[i]]
            ft_lab = label_names[ft_pred[i]]

            # Highlight switches
            if true_labels[i] in [2, 3]:
                token = f"*{token}*"

            # Check if predictions match truth
            binary_match = "✓" if binary_pred[i] == true_labels[i] else "✗"
            ft_match = "✓" if ft_pred[i] == true_labels[i] else "✗"

            print(f"[{i:3d}] {token:<20} {true_lab:<8} {binary_lab:<8}{binary_match} {ft_lab:<12}{ft_match}")

        # Show switch regions specifically
        switch_positions = [i for i, l in enumerate(true_labels) if l in [2, 3]]
        if switch_positions and len(switch_positions) <= 10:
            print(f"\nDetailed view at switch points:")
            for switch_pos in switch_positions[:5]:  # Show first 5 switches
                print(f"\n  Switch at position {switch_pos}:")
                start = max(0, switch_pos - 2)
                end = min(len(tokens), switch_pos + 3)

                for pos in range(start, end):
                    marker = ">>>" if pos == switch_pos else "   "
                    token = tokens[pos][:15]
                    true_lab = label_names[true_labels[pos]]
                    binary_lab = label_names[binary_pred[pos]]
                    ft_lab = label_names[ft_pred[pos]]

                    print(f"    {marker} [{pos:3d}] {token:<15} True:{true_lab:<8} Bin:{binary_lab:<8} FT:{ft_lab}")


def print_fbeta_comparison(binary_metrics, finetuned_metrics):
    """Print comprehensive F-beta(2) comparison with per-class metrics (5-token tolerance)"""

    print("\n" + "=" * 120)
    print("COMPREHENSIVE F-BETA(2) AND PER-CLASS COMPARISON WITH 5-TOKEN TOLERANCE")
    print("=" * 120)

    # Overall metrics with tolerance
    print("\n" + "─" * 120)
    print("OVERALL METRICS (5-token tolerance)")
    print("─" * 120)
    print(f"{'Metric':<20} {'Binary Model':<25} {'ALTO BeRT':<25} {'Difference':<20}")
    print("-" * 120)

    # Overall F-beta(2)
    print(f"{'F-beta(2)':<20} {binary_metrics['proximity_fbeta2']:<25.3f} "
          f"{finetuned_metrics['proximity_fbeta2']:<25.3f} "
          f"{finetuned_metrics['proximity_fbeta2'] - binary_metrics['proximity_fbeta2']:+20.3f}")

    # Overall Precision and Recall
    print(f"{'Precision':<20} {binary_metrics['proximity_precision']:<25.3f} "
          f"{finetuned_metrics['proximity_precision']:<25.3f} "
          f"{finetuned_metrics['proximity_precision'] - binary_metrics['proximity_precision']:+20.3f}")

    print(f"{'Recall':<20} {binary_metrics['proximity_recall']:<25.3f} "
          f"{finetuned_metrics['proximity_recall']:<25.3f} "
          f"{finetuned_metrics['proximity_recall'] - binary_metrics['proximity_recall']:+20.3f}")

    print(f"{'F1':<20} {binary_metrics['proximity_f1']:<25.3f} "
          f"{finetuned_metrics['proximity_f1']:<25.3f} "
          f"{finetuned_metrics['proximity_f1'] - binary_metrics['proximity_f1']:+20.3f}")

    # Per-class metrics with tolerance
    print("\n" + "─" * 120)
    print("PER-CLASS METRICS: SWITCH→AUTO (with 5-token tolerance)")
    print("─" * 120)
    print(f"{'Metric':<20} {'Binary Model':<25} {'ALTO BeRT':<25} {'Difference':<20}")
    print("-" * 120)

    # Use the values directly from metrics dictionary OR calculate from counts
    # For Switch→Auto
    if 'to_auto_proximity_precision' in binary_metrics:
        b_auto_precision = binary_metrics['to_auto_proximity_precision']
    else:
        # Calculate from matched counts
        b_auto_precision = (binary_metrics.get('matched_to_auto', 0) /
                            binary_metrics['pred_to_auto'] if binary_metrics.get('pred_to_auto', 0) > 0 else 0)

    if 'to_auto_proximity_precision' in finetuned_metrics:
        f_auto_precision = finetuned_metrics['to_auto_proximity_precision']
    else:
        f_auto_precision = (finetuned_metrics.get('matched_to_auto', 0) /
                            finetuned_metrics['pred_to_auto'] if finetuned_metrics.get('pred_to_auto', 0) > 0 else 0)

    if 'to_auto_proximity_recall' in binary_metrics:
        b_auto_recall = binary_metrics['to_auto_proximity_recall']
    else:
        b_auto_recall = (binary_metrics.get('matched_to_auto', 0) /
                         binary_metrics['true_to_auto'] if binary_metrics.get('true_to_auto', 0) > 0 else 0)

    if 'to_auto_proximity_recall' in finetuned_metrics:
        f_auto_recall = finetuned_metrics['to_auto_proximity_recall']
    else:
        f_auto_recall = (finetuned_metrics.get('matched_to_auto', 0) /
                         finetuned_metrics['true_to_auto'] if finetuned_metrics.get('true_to_auto', 0) > 0 else 0)

    print(f"{'F-beta(2)':<20} {binary_metrics.get('to_auto_proximity_fbeta2', 0):<25.3f} "
          f"{finetuned_metrics.get('to_auto_proximity_fbeta2', 0):<25.3f} "
          f"{finetuned_metrics.get('to_auto_proximity_fbeta2', 0) - binary_metrics.get('to_auto_proximity_fbeta2', 0):+20.3f}")

    print(f"{'Precision':<20} {b_auto_precision:<25.3f} "
          f"{f_auto_precision:<25.3f} "
          f"{f_auto_precision - b_auto_precision:+20.3f}")

    print(f"{'Recall':<20} {b_auto_recall:<25.3f} "
          f"{f_auto_recall:<25.3f} "
          f"{f_auto_recall - b_auto_recall:+20.3f}")

    # F1 for Switch→Auto
    b_auto_f1 = 2 * b_auto_precision * b_auto_recall / (b_auto_precision + b_auto_recall) if (
                                                                                                         b_auto_precision + b_auto_recall) > 0 else 0
    f_auto_f1 = 2 * f_auto_precision * f_auto_recall / (f_auto_precision + f_auto_recall) if (
                                                                                                         f_auto_precision + f_auto_recall) > 0 else 0

    print(f"{'F1':<20} {b_auto_f1:<25.3f} "
          f"{f_auto_f1:<25.3f} "
          f"{f_auto_f1 - b_auto_f1:+20.3f}")

    print(f"{'Support (count)':<20} {binary_metrics.get('true_to_auto', 0):<25} "
          f"{finetuned_metrics.get('true_to_auto', 0):<25} "
          f"{'(same test set)':<20}")

    print("\n" + "─" * 120)
    print("PER-CLASS METRICS: SWITCH→ALLO (with 5-token tolerance)")
    print("─" * 120)
    print(f"{'Metric':<20} {'Binary Model':<25} {'ALTO BeRT':<25} {'Difference':<20}")
    print("-" * 120)

    # For Switch→Allo
    if 'to_allo_proximity_precision' in binary_metrics:
        b_allo_precision = binary_metrics['to_allo_proximity_precision']
    else:
        b_allo_precision = (binary_metrics.get('matched_to_allo', 0) /
                            binary_metrics['pred_to_allo'] if binary_metrics.get('pred_to_allo', 0) > 0 else 0)

    if 'to_allo_proximity_precision' in finetuned_metrics:
        f_allo_precision = finetuned_metrics['to_allo_proximity_precision']
    else:
        f_allo_precision = (finetuned_metrics.get('matched_to_allo', 0) /
                            finetuned_metrics['pred_to_allo'] if finetuned_metrics.get('pred_to_allo', 0) > 0 else 0)

    if 'to_allo_proximity_recall' in binary_metrics:
        b_allo_recall = binary_metrics['to_allo_proximity_recall']
    else:
        b_allo_recall = (binary_metrics.get('matched_to_allo', 0) /
                         binary_metrics['true_to_allo'] if binary_metrics.get('true_to_allo', 0) > 0 else 0)

    if 'to_allo_proximity_recall' in finetuned_metrics:
        f_allo_recall = finetuned_metrics['to_allo_proximity_recall']
    else:
        f_allo_recall = (finetuned_metrics.get('matched_to_allo', 0) /
                         finetuned_metrics['true_to_allo'] if finetuned_metrics.get('true_to_allo', 0) > 0 else 0)

    print(f"{'F-beta(2)':<20} {binary_metrics.get('to_allo_proximity_fbeta2', 0):<25.3f} "
          f"{finetuned_metrics.get('to_allo_proximity_fbeta2', 0):<25.3f} "
          f"{finetuned_metrics.get('to_allo_proximity_fbeta2', 0) - binary_metrics.get('to_allo_proximity_fbeta2', 0):+20.3f}")

    print(f"{'Precision':<20} {b_allo_precision:<25.3f} "
          f"{f_allo_precision:<25.3f} "
          f"{f_allo_precision - b_allo_precision:+20.3f}")

    print(f"{'Recall':<20} {b_allo_recall:<25.3f} "
          f"{f_allo_recall:<25.3f} "
          f"{f_allo_recall - b_allo_recall:+20.3f}")

    # F1 for Switch→Allo
    b_allo_f1 = 2 * b_allo_precision * b_allo_recall / (b_allo_precision + b_allo_recall) if (
                                                                                                         b_allo_precision + b_allo_recall) > 0 else 0
    f_allo_f1 = 2 * f_allo_precision * f_allo_recall / (f_allo_precision + f_allo_recall) if (
                                                                                                         f_allo_precision + f_allo_recall) > 0 else 0

    print(f"{'F1':<20} {b_allo_f1:<25.3f} "
          f"{f_allo_f1:<25.3f} "
          f"{f_allo_f1 - b_allo_f1:+20.3f}")

    print(f"{'Support (count)':<20} {binary_metrics.get('true_to_allo', 0):<25} "
          f"{finetuned_metrics.get('true_to_allo', 0):<25} "
          f"{'(same test set)':<20}")



def show_detailed_segment_comparisons(test_df, binary_tokenizer, binary_session, ft_tokenizer, ft_model,
                                      num_examples=10):
    """Show detailed side-by-side comparisons with actual label names"""

    device = next(ft_model.parameters()).device

    # Label names for display
    label_names = {
        0: 'Auto',
        1: 'Allo',
        2: '→AUTO',
        3: '→ALLO'
    }

    print("\n" + "=" * 120)
    print("DETAILED SEGMENT-BY-SEGMENT COMPARISON")
    print("=" * 120)

    # Sample random segments
    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    for ex_num, idx in enumerate(sample_indices, 1):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()
        true_labels = [int(l) for l in row['labels'].split(',')]

        # Get predictions
        binary_pred = process_binary_model_sentence_level(tokens, binary_tokenizer, binary_session)
        ft_pred = process_finetuned_model(tokens, ft_tokenizer, ft_model, device)

        # Align lengths
        min_len = min(len(tokens), len(true_labels), len(binary_pred), len(ft_pred))
        tokens = tokens[:min_len]
        true_labels = true_labels[:min_len]
        binary_pred = binary_pred[:min_len]
        ft_pred = ft_pred[:min_len]

        print(f"\n{'─' * 120}")
        print(f"SEGMENT {ex_num} | File: {row['source_file'][:60]}...")
        print(f"Length: {len(tokens)} tokens | True switches: {sum(1 for l in true_labels if l in [2, 3])}")
        print(f"{'─' * 120}")

        # Calculate metrics for this segment
        seg_binary_metrics = evaluate_switch_detection_with_proximity(true_labels, binary_pred, tolerance=5)
        seg_ft_metrics = evaluate_switch_detection_with_proximity(true_labels, ft_pred, tolerance=5)

        print(f"\nSegment Metrics:")
        print(f"  Binary Model - F-beta(2): {seg_binary_metrics['proximity_fbeta2']:.3f} | "
              f"Precision: {seg_binary_metrics['proximity_precision']:.3f} | "
              f"Recall: {seg_binary_metrics['proximity_recall']:.3f}")
        print(f"  Fine-tuned   - F-beta(2): {seg_ft_metrics['proximity_fbeta2']:.3f} | "
              f"Precision: {seg_ft_metrics['proximity_precision']:.3f} | "
              f"Recall: {seg_ft_metrics['proximity_recall']:.3f}")

        # Show detailed comparison
        print(f"\nToken-by-token comparison (showing first 40 tokens):")
        print(f"{'─' * 120}")
        print(f"{'Pos':<5} {'Token':<15} {'True Label':<10} {'Binary Pred':<12} {'Fine-tuned':<12} {'Match'}")
        print(f"{'─' * 120}")

        for i in range(min(40, len(tokens))):
            token = tokens[i][:14]
            true_label = label_names[true_labels[i]]
            binary_label = label_names[binary_pred[i]]
            ft_label = label_names[ft_pred[i]]

            # Check matches
            binary_match = "✓" if binary_pred[i] == true_labels[i] else "✗"
            ft_match = "✓" if ft_pred[i] == true_labels[i] else "✓"

            # Highlight switch points
            if true_labels[i] in [2, 3]:
                print(
                    f"[{i:3d}] {token:<15} **{true_label:<10} {binary_label:<10}{binary_match}  {ft_label:<10}{ft_match}")
            else:
                print(
                    f"[{i:3d}] {token:<15} {true_label:<10} {binary_label:<10}{binary_match}  {ft_label:<10}{ft_match}")

        # Show switch regions in detail
        true_switches = [(i, true_labels[i]) for i in range(len(true_labels)) if true_labels[i] in [2, 3]]

        if true_switches:
            print(f"\n{'─' * 120}")
            print(f"DETAILED VIEW AT SWITCH POINTS:")
            print(f"{'─' * 120}")

            for switch_idx, switch_type in true_switches[:5]:  # Show first 5 switches
                print(f"\nSwitch at position {switch_idx}: {label_names[switch_type]}")

                start = max(0, switch_idx - 3)
                end = min(len(tokens), switch_idx + 4)

                print(f"{'Pos':<8} {'Token':<15} {'True':<10} {'Binary':<10} {'Fine-tuned':<10}")
                print("-" * 60)

                for pos in range(start, end):
                    marker = ">>>" if pos == switch_idx else "   "
                    token = tokens[pos][:14]
                    true_label = label_names[true_labels[pos]]
                    binary_label = label_names[binary_pred[pos]]
                    ft_label = label_names[ft_pred[pos]]

                    if pos == switch_idx:
                        # Highlight the switch point
                        print(f"{marker} [{pos:3d}] {token:<15} {true_label:<10} {binary_label:<10} {ft_label:<10}")
                    else:
                        print(f"    [{pos:3d}] {token:<15} {true_label:<10} {binary_label:<10} {ft_label:<10}")

        # Summary for this segment
        print(f"\n{'─' * 120}")
        print(f"Summary for Segment {ex_num}:")
        true_auto_switches = sum(1 for l in true_labels if l == 2)
        true_allo_switches = sum(1 for l in true_labels if l == 3)
        binary_auto_switches = sum(1 for l in binary_pred if l == 2)
        binary_allo_switches = sum(1 for l in binary_pred if l == 3)
        ft_auto_switches = sum(1 for l in ft_pred if l == 2)
        ft_allo_switches = sum(1 for l in ft_pred if l == 3)

        print(f"  True Labels:    {true_auto_switches} →AUTO, {true_allo_switches} →ALLO")
        print(f"  Binary Model:   {binary_auto_switches} →AUTO, {binary_allo_switches} →ALLO")
        print(f"  Fine-tuned:     {ft_auto_switches} →AUTO, {ft_allo_switches} →ALLO")


if __name__ == "__main__":
    binary_metrics, finetuned_metrics = unified_evaluation()

    print_fbeta_comparison(binary_metrics, finetuned_metrics)

    # Load models for detailed comparison
    print("\n\nLoading models for detailed segment comparisons...")

    TEST_FILE = './dataset/annotated-data/test_segments.csv'
    test_df = pd.read_csv(TEST_FILE)


    binary_tokenizer = AutoTokenizer.from_pretrained('./alloauto-presentation/web/model')
    binary_session = ort.InferenceSession('./alloauto-presentation/web/model/onnx/model.onnx')

    model_id = "levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_all_data_no_labels_no_partial"
    # model_id = "levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_all_data"
    ft_tokenizer = AutoTokenizer.from_pretrained(model_id)
    ft_model = AutoModelForTokenClassification.from_pretrained(model_id)
    ft_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_model = ft_model.to(device)

    # Show detailed comparisons
    show_detailed_segment_comparisons(test_df, binary_tokenizer, binary_session,
                                      ft_tokenizer, ft_model, num_examples=1)
