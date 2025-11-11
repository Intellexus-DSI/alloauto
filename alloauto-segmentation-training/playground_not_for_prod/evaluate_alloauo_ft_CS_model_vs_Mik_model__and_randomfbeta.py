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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def process_random_model(tokens, avg_switches_per_segment=3.5, seed=42):
    """
    Random baseline: assigns random switch points
    CAN predict 0 switches if that's what the random draw gives
    """
    np.random.seed(seed)

    n_tokens = len(tokens)

    # Allow 0 switches - just use Poisson directly
    n_switches = int(np.random.poisson(avg_switches_per_segment))

    # If 0 switches, entire segment is one mode
    if n_switches == 0:
        # Randomly pick a single mode for entire segment
        mode = np.random.choice([0, 1])  # All Auto or all Allo
        return [mode] * n_tokens

    # Rest of the code stays the same...
    possible_positions = list(range(10, n_tokens - 10))
    if len(possible_positions) < n_switches:
        n_switches = len(possible_positions)

    if n_switches == 0 or len(possible_positions) == 0:
        # Edge case: segment too short for switches
        mode = np.random.choice([0, 1])
        return [mode] * n_tokens

    switch_positions = sorted(np.random.choice(possible_positions, n_switches, replace=False))

    # Generate labels with random switches
    labels = []
    current_mode = np.random.choice([0, 1])  # Start randomly
    switch_idx = 0

    for i in range(n_tokens):
        if switch_idx < len(switch_positions) and i == switch_positions[switch_idx]:
            # Switch position
            if current_mode == 0:
                labels.append(3)  # Switch to Allo
                current_mode = 1
            else:
                labels.append(2)  # Switch to Auto
                current_mode = 0
            switch_idx += 1
        else:
            # Non-switch position
            labels.append(current_mode)

    return labels
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

from fine_tune_CS_4_classes_clean_no_allo_auto_labels_CRF import apply_transition_constraints
def process_alto_with_constraints(tokens, tokenizer, model, device):
    """ALTO model with post-processing constraints (no CRF)"""
    # Get raw predictions from ALTO
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
    aligned_preds = []
    previous_word_idx = None

    for j, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            aligned_preds.append(predictions[0][j].item())
        previous_word_idx = word_idx

    # Apply logical constraints (this function already exists in your code)
    constrained_preds = apply_transition_constraints(aligned_preds)

    return constrained_preds
def unified_evaluation():
    """Main evaluation function on complete test set - now includes random baseline"""
    print("=" * 80)
    print("VERIFYING DATASETS ARE TAG-FREE")
    print("=" * 80)

    datasets_clean = verify_no_tags_in_datasets()
    if not datasets_clean:
        print("\n⚠️ ERROR: Tags found in datasets!")
        print("Please re-run preprocessing to remove tags.")
        return None, None, None  # Now returns 3 values

    TEST_FILE = './test_segments.csv'  # UPDATE THIS

    # TEST_FILE = './dataset/annotated-data/test_segments.csv'
    TOLERANCE = 5

    print(f"Loading test data from: {TEST_FILE}")
    test_df = pd.read_csv(TEST_FILE)
    print(f"Test set size: {len(test_df)} segments")

    # Calculate test set statistics for random model
    total_tokens = 0
    total_switches = 0
    for idx in range(len(test_df)):
        labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
        total_tokens += len(labels)
        total_switches += sum(1 for l in labels if l in [2, 3])

    avg_switches = total_switches / len(test_df)
    print(f"Total tokens in test set: {total_tokens}")
    print(f"Total switches in test set: {total_switches}")
    print(f"Average switches per segment: {avg_switches:.2f}\n")

    # Load Binary Model
    print("Loading Binary Model...")
    binary_tokenizer = AutoTokenizer.from_pretrained('./alloauto-presentation/web/model')
    binary_session = ort.InferenceSession('./alloauto-presentation/web/model/onnx/model.onnx')

    print("Loading mBERT Baseline...")
    mbert_tokenizer = AutoTokenizer.from_pretrained('./alloauto-segmentation-training/benchmark_models/mbert_baseline_model/final_model')
    mbert_model = AutoModelForTokenClassification.from_pretrained('./alloauto-segmentation-training/benchmark_models/mbert_baseline_model/final_model')
    mbert_model.eval()

    print("Loading XLM-RoBERTa Baseline...")
    xlmr_tokenizer = AutoTokenizer.from_pretrained('./alloauto-segmentation-training/benchmark_models/xlmroberta_baseline_model/final_model')
    xlmr_model = AutoModelForTokenClassification.from_pretrained('./alloauto-segmentation-training/benchmark_models/xlmroberta_baseline_model/final_model')
    xlmr_model.eval()

    print("Loading CRF Model...")
    from fine_tune_CS_4_classes_clean_no_allo_auto_labels_CRF import BERTWithCRFWrapper

    crf_tokenizer = AutoTokenizer.from_pretrained('./alloauto-segmentation-training/fine_tuned_ALTO_models/crf_enhanced_model')
    crf_model = BERTWithCRFWrapper.from_pretrained('./alloauto-segmentation-training/fine_tuned_ALTO_models/crf_for_ALTO_allow_non_switch_test_train_and_fixed_loss_6_10')
    # crf_model = BERTWithCRFWrapper.from_pretrained('./alloauto-segmentation-training/fine_tuned_ALTO_models/crf_enhanced_model')
    crf_model.eval()

    # Load Fine-tuned Model
    print("Loading Fine-tuned Model...")
    model_id = "./alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_allow_non_switch_test_train_and_fixed_loss_6_10/final_model"
    # model_id = "./tibetan_code_switching_constrained_model_wylie-final_all_data_no_labels_no_prtial_v2_2_10/final_model"
    ft_tokenizer = AutoTokenizer.from_pretrained(model_id)
    ft_model = AutoModelForTokenClassification.from_pretrained(model_id)
    ft_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mbert_model = mbert_model.to(device)
    xlmr_model = xlmr_model.to(device)
    ft_model = ft_model.to(device)
    crf_model = crf_model.to(device)  # NEW

    print(f"Using device: {device}\n")

    # Process all test segments
    print("Processing all test segments...")
    all_true_labels = []
    binary_all_pred = []
    finetuned_all_pred = []
    mbert_all_pred = []
    random_all_pred = []  # NEW
    crf_all_pred = []
    xlmr_all_pred = []

    for idx, row in test_df.iterrows():
        if idx % 20 == 0:
            print(f"  Processing segment {idx}/{len(test_df)}...")

        tokens = row['tokens'].split()
        true_labels = [int(l) for l in row['labels'].split(',')]

        # Get all predictions
        random_pred = process_random_model(tokens, avg_switches_per_segment=avg_switches, seed=42 + idx)
        binary_pred = process_binary_model_sentence_level(tokens, binary_tokenizer, binary_session)
        mbert_pred = process_finetuned_model(tokens, mbert_tokenizer, mbert_model, device)
        xlmr_pred = process_finetuned_model(tokens, xlmr_tokenizer, xlmr_model, device)  # NEW
        crf_pred = process_alto_with_constraints(tokens, ft_tokenizer, ft_model, device)

        alto_pred = process_finetuned_model(tokens, ft_tokenizer, ft_model, device)

        # Align lengths
        min_len = min(len(true_labels), len(random_pred), len(binary_pred),
                      len(mbert_pred), len(xlmr_pred), len(alto_pred), len(crf_pred))

        all_true_labels.extend(true_labels[:min_len])
        random_all_pred.extend(random_pred[:min_len])
        binary_all_pred.extend(binary_pred[:min_len])
        mbert_all_pred.extend(mbert_pred[:min_len])
        xlmr_all_pred.extend(xlmr_pred[:min_len])  # NEW
        finetuned_all_pred.extend(alto_pred[:min_len])
        crf_all_pred.extend(crf_pred[:min_len])  # NEW

        # Calculate metrics
    random_metrics = evaluate_switch_detection_with_proximity(all_true_labels, random_all_pred, TOLERANCE)
    binary_metrics = evaluate_switch_detection_with_proximity(all_true_labels, binary_all_pred, TOLERANCE)
    mbert_metrics = evaluate_switch_detection_with_proximity(all_true_labels, mbert_all_pred, TOLERANCE)
    xlmr_metrics = evaluate_switch_detection_with_proximity(all_true_labels, xlmr_all_pred, TOLERANCE)  # NEW
    alto_metrics = evaluate_switch_detection_with_proximity(all_true_labels, finetuned_all_pred, TOLERANCE)
    crf_metrics = evaluate_switch_detection_with_proximity(all_true_labels, crf_all_pred, TOLERANCE)  # NEW
    print(f"\n{'=' * 180}")
    print("6-MODEL COMPARISON")
    print(f"{'=' * 180}")
    print(f"{'Metric':<30} {'Random':<16} {'Binary':<16} {'mBERT':<16} {'XLM-R':<16} {'ALTO':<16} {'CRF':<16}")
    print("-" * 180)

    metrics_to_show = [
        ('F-beta(2)', 'proximity_fbeta2'),
        ('Precision', 'proximity_precision'),
        ('Recall', 'proximity_recall'),
    ]

    for display, key in metrics_to_show:
        r = random_metrics[key]
        b = binary_metrics[key]
        m = mbert_metrics[key]
        x = xlmr_metrics[key]
        a = alto_metrics[key]
        c = crf_metrics[key]  # NEW

        print(f"{display:<30} {r:<16.3f} {b:<16.3f} {m:<16.3f} {x:<16.3f} {a:<16.3f} {c:<16.3f}")

    random_metrics = evaluate_switch_detection_with_proximity(all_true_labels, random_all_pred, TOLERANCE)
    binary_metrics = evaluate_switch_detection_with_proximity(all_true_labels, binary_all_pred, TOLERANCE)
    mbert_metrics = evaluate_switch_detection_with_proximity(all_true_labels, mbert_all_pred, TOLERANCE)
    xlmr_metrics = evaluate_switch_detection_with_proximity(all_true_labels, xlmr_all_pred, TOLERANCE)
    alto_metrics = evaluate_switch_detection_with_proximity(all_true_labels, finetuned_all_pred, TOLERANCE)
    crf_metrics = evaluate_switch_detection_with_proximity(all_true_labels, crf_all_pred, TOLERANCE)

    # Print overall comparison (existing code)
    print(f"\n{'=' * 180}")
    print("6-MODEL COMPARISON")
    # ... existing printing code ...

    # NEW: Analyze by segment type
    model_predictions_dict = {
        'Random': random_all_pred,
        'Binary': binary_all_pred,
        'mBERT': mbert_all_pred,
        'XLM-R': xlmr_all_pred,
        'ALTO': finetuned_all_pred,
        'CRF': crf_all_pred
    }

    segment_type_results = analyze_performance_by_segment_type(
        test_df, all_true_labels, model_predictions_dict, TOLERANCE
    )

    return random_metrics, binary_metrics, mbert_metrics, xlmr_metrics, alto_metrics, crf_metrics, segment_type_results

def unified_evaluation_old():
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
    TEST_FILE = './test_segments.csv'
    # TEST_FILE = './dataset/annotated-data/test_segments.csv'
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
                                      num_examples=10, avg_switches=3.5):
    """Show detailed side-by-side comparisons with actual label names - including random baseline"""

    device = next(ft_model.parameters()).device

    # Label names for display
    label_names = {
        0: 'Auto',
        1: 'Allo',
        2: '→AUTO',
        3: '→ALLO'
    }

    print("\n" + "=" * 140)
    print("DETAILED SEGMENT-BY-SEGMENT COMPARISON (INCLUDING RANDOM BASELINE)")
    print("=" * 140)

    # Sample random segments
    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    for ex_num, idx in enumerate(sample_indices, 1):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()
        true_labels = [int(l) for l in row['labels'].split(',')]

        # Get predictions from all three models
        binary_pred = process_binary_model_sentence_level(tokens, binary_tokenizer, binary_session)
        ft_pred = process_finetuned_model(tokens, ft_tokenizer, ft_model, device)
        random_pred = process_random_model(tokens, avg_switches_per_segment=avg_switches, seed=42+idx)

        # Align lengths
        min_len = min(len(tokens), len(true_labels), len(binary_pred), len(ft_pred), len(random_pred))
        tokens = tokens[:min_len]
        true_labels = true_labels[:min_len]
        binary_pred = binary_pred[:min_len]
        ft_pred = ft_pred[:min_len]
        random_pred = random_pred[:min_len]

        print(f"\n{'─' * 140}")
        print(f"SEGMENT {ex_num} | File: {row['source_file'][:60]}...")
        print(f"Length: {len(tokens)} tokens | True switches: {sum(1 for l in true_labels if l in [2, 3])}")
        print(f"{'─' * 140}")

        # Calculate metrics for this segment
        seg_random_metrics = evaluate_switch_detection_with_proximity(true_labels, random_pred, tolerance=5)
        seg_binary_metrics = evaluate_switch_detection_with_proximity(true_labels, binary_pred, tolerance=5)
        seg_ft_metrics = evaluate_switch_detection_with_proximity(true_labels, ft_pred, tolerance=5)

        print(f"\nSegment Metrics:")
        print(f"  Random Model - F-beta(2): {seg_random_metrics['proximity_fbeta2']:.3f} | "
              f"Precision: {seg_random_metrics['proximity_precision']:.3f} | "
              f"Recall: {seg_random_metrics['proximity_recall']:.3f}")
        print(f"  Binary Model - F-beta(2): {seg_binary_metrics['proximity_fbeta2']:.3f} | "
              f"Precision: {seg_binary_metrics['proximity_precision']:.3f} | "
              f"Recall: {seg_binary_metrics['proximity_recall']:.3f}")
        print(f"  Fine-tuned   - F-beta(2): {seg_ft_metrics['proximity_fbeta2']:.3f} | "
              f"Precision: {seg_ft_metrics['proximity_precision']:.3f} | "
              f"Recall: {seg_ft_metrics['proximity_recall']:.3f}")

        # Show detailed comparison
        print(f"\nToken-by-token comparison (showing first 40 tokens):")
        print(f"{'─' * 140}")
        print(f"{'Pos':<5} {'Token':<15} {'True':<8} {'Random':<10} {'Binary':<10} {'Fine-tuned':<12}")
        print(f"{'─' * 140}")

        for i in range(min(40, len(tokens))):
            token = tokens[i][:14]
            true_label = label_names[true_labels[i]]
            random_label = label_names[random_pred[i]]
            binary_label = label_names[binary_pred[i]]
            ft_label = label_names[ft_pred[i]]

            # Check matches
            random_match = "✓" if random_pred[i] == true_labels[i] else "✗"
            binary_match = "✓" if binary_pred[i] == true_labels[i] else "✗"
            ft_match = "✓" if ft_pred[i] == true_labels[i] else "✗"

            # Highlight switch points
            if true_labels[i] in [2, 3]:
                print(f"[{i:3d}] {token:<15} **{true_label:<8} "
                      f"{random_label:<8}{random_match}  "
                      f"{binary_label:<8}{binary_match}  "
                      f"{ft_label:<10}{ft_match}")
            else:
                print(f"[{i:3d}] {token:<15} {true_label:<8} "
                      f"{random_label:<8}{random_match}  "
                      f"{binary_label:<8}{binary_match}  "
                      f"{ft_label:<10}{ft_match}")

        # Show switch regions in detail
        true_switches = [(i, true_labels[i]) for i in range(len(true_labels)) if true_labels[i] in [2, 3]]

        if true_switches:
            print(f"\n{'─' * 140}")
            print(f"DETAILED VIEW AT SWITCH POINTS:")
            print(f"{'─' * 140}")

            for switch_idx, switch_type in true_switches[:5]:  # Show first 5 switches
                print(f"\nSwitch at position {switch_idx}: {label_names[switch_type]}")

                start = max(0, switch_idx - 3)
                end = min(len(tokens), switch_idx + 4)

                print(f"{'Pos':<8} {'Token':<15} {'True':<10} {'Random':<10} {'Binary':<10} {'Fine-tuned':<10}")
                print("-" * 80)

                for pos in range(start, end):
                    marker = ">>>" if pos == switch_idx else "   "
                    token = tokens[pos][:14]
                    true_label = label_names[true_labels[pos]]
                    random_label = label_names[random_pred[pos]]
                    binary_label = label_names[binary_pred[pos]]
                    ft_label = label_names[ft_pred[pos]]

                    if pos == switch_idx:
                        # Highlight the switch point
                        print(f"{marker} [{pos:3d}] {token:<15} {true_label:<10} {random_label:<10} {binary_label:<10} {ft_label:<10}")
                    else:
                        print(f"    [{pos:3d}] {token:<15} {true_label:<10} {random_label:<10} {binary_label:<10} {ft_label:<10}")

        # Summary for this segment
        print(f"\n{'─' * 140}")
        print(f"Summary for Segment {ex_num}:")
        true_auto_switches = sum(1 for l in true_labels if l == 2)
        true_allo_switches = sum(1 for l in true_labels if l == 3)
        random_auto_switches = sum(1 for l in random_pred if l == 2)
        random_allo_switches = sum(1 for l in random_pred if l == 3)
        binary_auto_switches = sum(1 for l in binary_pred if l == 2)
        binary_allo_switches = sum(1 for l in binary_pred if l == 3)
        ft_auto_switches = sum(1 for l in ft_pred if l == 2)
        ft_allo_switches = sum(1 for l in ft_pred if l == 3)

        print(f"  True Labels:    {true_auto_switches} →AUTO, {true_allo_switches} →ALLO")
        print(f"  Random Model:   {random_auto_switches} →AUTO, {random_allo_switches} →ALLO")
        print(f"  Binary Model:   {binary_auto_switches} →AUTO, {binary_allo_switches} →ALLO")
        print(f"  Fine-tuned:     {ft_auto_switches} →AUTO, {ft_allo_switches} →ALLO")
def show_detailed_segment_comparisons_old(test_df, binary_tokenizer, binary_session, ft_tokenizer, ft_model,
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


def print_fbeta_comparison_three_models(random_metrics, binary_metrics, finetuned_metrics):
    """Print comprehensive comparison including random baseline"""

    print("\n" + "=" * 120)
    print("THREE-WAY MODEL COMPARISON")
    print("=" * 120)

    print(f"\n{'Metric':<35} {'Random':<20} {'Binary':<20} {'ALTO BeRT':<20}")
    print("-" * 120)

    metrics_to_show = [
        ('Overall F-beta(2)', 'proximity_fbeta2'),
        ('Overall Precision', 'proximity_precision'),
        ('Overall Recall', 'proximity_recall'),
        ('Macro F-beta(2)', 'macro_fbeta2'),
        ('Switch→Auto F-beta(2)', 'to_auto_fbeta2'),
        ('Switch→Allo F-beta(2)', 'to_allo_fbeta2'),
    ]

    for display, key in metrics_to_show:
        r_val = random_metrics[key]
        b_val = binary_metrics[key]
        f_val = finetuned_metrics[key]

        # Highlight best
        best_val = max(r_val, b_val, f_val)
        r_str = f"{r_val:.3f}" + (" ★" if r_val == best_val else "")
        b_str = f"{b_val:.3f}" + (" ★" if b_val == best_val else "")
        f_str = f"{f_val:.3f}" + (" ★" if f_val == best_val else "")

        print(f"{display:<35} {r_str:<20} {b_str:<20} {f_str:<20}")


def show_detailed_segment_comparisons_four_models(test_df, binary_tokenizer, binary_session,
                                                  mbert_tokenizer, mbert_model,
                                                  alto_tokenizer, alto_model,
                                                  num_examples=3, avg_switches=3.5):
    """Show detailed comparisons with all 4 models"""

    device = next(alto_model.parameters()).device

    label_names = {0: 'Auto', 1: 'Allo', 2: '→AUTO', 3: '→ALLO'}

    print("\n" + "=" * 160)
    print("DETAILED SEGMENT COMPARISON (4 MODELS)")
    print("=" * 160)

    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    for ex_num, idx in enumerate(sample_indices, 1):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()
        true_labels = [int(l) for l in row['labels'].split(',')]

        # Get predictions from all 4 models
        random_pred = process_random_model(tokens, avg_switches_per_segment=avg_switches, seed=42 + idx)
        binary_pred = process_binary_model_sentence_level(tokens, binary_tokenizer, binary_session)
        mbert_pred = process_finetuned_model(tokens, mbert_tokenizer, mbert_model, device)
        alto_pred = process_finetuned_model(tokens, alto_tokenizer, alto_model, device)

        # Align lengths
        min_len = min(len(tokens), len(true_labels), len(random_pred),
                      len(binary_pred), len(mbert_pred), len(alto_pred))
        tokens = tokens[:min_len]
        true_labels = true_labels[:min_len]
        random_pred = random_pred[:min_len]
        binary_pred = binary_pred[:min_len]
        mbert_pred = mbert_pred[:min_len]
        alto_pred = alto_pred[:min_len]

        print(f"\n{'─' * 160}")
        print(f"SEGMENT {ex_num} | File: {row['source_file'][:60]}...")
        print(f"Length: {len(tokens)} tokens | True switches: {sum(1 for l in true_labels if l in [2, 3])}")
        print(f"{'─' * 160}")

        # Calculate metrics
        seg_random = evaluate_switch_detection_with_proximity(true_labels, random_pred, tolerance=5)
        seg_binary = evaluate_switch_detection_with_proximity(true_labels, binary_pred, tolerance=5)
        seg_mbert = evaluate_switch_detection_with_proximity(true_labels, mbert_pred, tolerance=5)
        seg_alto = evaluate_switch_detection_with_proximity(true_labels, alto_pred, tolerance=5)

        print(f"\nSegment Metrics (F-beta(2)):")
        print(f"  Random:   {seg_random['proximity_fbeta2']:.3f}")
        print(f"  Binary:   {seg_binary['proximity_fbeta2']:.3f}")
        print(f"  mBERT:    {seg_mbert['proximity_fbeta2']:.3f}")
        print(f"  ALTO:     {seg_alto['proximity_fbeta2']:.3f}")

        # Show first 30 tokens
        print(f"\nToken-by-token (first 30 tokens):")
        print(f"{'─' * 160}")
        print(f"{'Pos':<5} {'Token':<15} {'True':<8} {'Random':<8} {'Binary':<8} {'mBERT':<8} {'ALTO':<8}")
        print(f"{'─' * 160}")

        for i in range(min(30, len(tokens))):
            token = tokens[i][:14]
            true_label = label_names[true_labels[i]]
            random_label = label_names[random_pred[i]]
            binary_label = label_names[binary_pred[i]]
            mbert_label = label_names[mbert_pred[i]]
            alto_label = label_names[alto_pred[i]]

            # Highlight switches
            if true_labels[i] in [2, 3]:
                print(
                    f"[{i:3d}] {token:<15} **{true_label:<8} {random_label:<8} {binary_label:<8} {mbert_label:<8} {alto_label:<8}")
            else:
                print(
                    f"[{i:3d}] {token:<15} {true_label:<8} {random_label:<8} {binary_label:<8} {mbert_label:<8} {alto_label:<8}")

        # Summary
        print(f"\n{'─' * 160}")
        print(f"Summary:")
        print(
            f"  True:     {sum(1 for l in true_labels if l == 2)} →AUTO, {sum(1 for l in true_labels if l == 3)} →ALLO")
        print(
            f"  Random:   {sum(1 for l in random_pred if l == 2)} →AUTO, {sum(1 for l in random_pred if l == 3)} →ALLO")
        print(
            f"  Binary:   {sum(1 for l in binary_pred if l == 2)} →AUTO, {sum(1 for l in binary_pred if l == 3)} →ALLO")
        print(f"  mBERT:    {sum(1 for l in mbert_pred if l == 2)} →AUTO, {sum(1 for l in mbert_pred if l == 3)} →ALLO")
        print(f"  ALTO:     {sum(1 for l in alto_pred if l == 2)} →AUTO, {sum(1 for l in alto_pred if l == 3)} →ALLO")
def print_fbeta_comparison_four_models(random_metrics, binary_metrics, mbert_metrics, alto_metrics):
    """Print comprehensive comparison of all 4 models"""

    print("\n" + "=" * 140)
    print("FOUR-WAY MODEL COMPARISON (5-token tolerance)")
    print("=" * 140)

    print(f"\n{'Metric':<35} {'Random':<20} {'Binary':<20} {'mBERT':<20} {'ALTO-BERT':<20}")
    print("-" * 140)

    metrics_to_show = [
        ('F-beta(2)', 'proximity_fbeta2'),
        ('Precision', 'proximity_precision'),
        ('Recall', 'proximity_recall'),
        ('F1', 'proximity_f1'),
        ('Macro F-beta(2)', 'macro_fbeta2'),
        ('Switch→Auto F-beta(2)', 'to_auto_fbeta2'),
        ('Switch→Allo F-beta(2)', 'to_allo_fbeta2'),
    ]

    for display, key in metrics_to_show:
        r = random_metrics[key]
        b = binary_metrics[key]
        m = mbert_metrics[key]
        a = alto_metrics[key]

        # Mark the best (excluding random)
        best_val = max(b, m, a)
        b_mark = " ★" if b == best_val else ""
        m_mark = " ★" if m == best_val else ""
        a_mark = " ★" if a == best_val else ""

        print(f"{display:<35} {r:<20.3f} {b:<20.3f}{b_mark:<4} {m:<20.3f}{m_mark:<4} {a:<20.3f}{a_mark:<4}")

    print("\n" + "=" * 140)
    print("IMPROVEMENT OVER RANDOM BASELINE")
    print("=" * 140)

    key = 'proximity_fbeta2'
    baseline = random_metrics[key]

    print(f"{'Model':<30} {'F-beta(2)':<15} {'vs Random':<20}")
    print("-" * 65)
    print(f"{'Random Baseline':<30} {baseline:<15.3f} {'—':<20}")
    print(
        f"{'Binary':<30} {binary_metrics[key]:<15.3f} {f'+{(binary_metrics[key] - baseline) / baseline * 100:.1f}%':<20}")
    print(
        f"{'mBERT':<30} {mbert_metrics[key]:<15.3f} {f'+{(mbert_metrics[key] - baseline) / baseline * 100:.1f}%':<20}")
    print(
        f"{'ALTO-BERT':<30} {alto_metrics[key]:<15.3f} {f'+{(alto_metrics[key] - baseline) / baseline * 100:.1f}%':<20}")

    print("\n" + "=" * 140)
    print("KEY FINDING")
    print("=" * 140)

    if alto_metrics[key] > mbert_metrics[key]:
        improvement = (alto_metrics[key] - mbert_metrics[key]) / mbert_metrics[key] * 100
        print(f"✓ Tibetan-specialized pretraining helps: ALTO-BERT outperforms mBERT by {improvement:.1f}%")
    else:
        print(f"⚠ Tibetan-specialized pretraining doesn't help: mBERT performs similarly or better")

def print_fbeta_comparison_five_models(random_metrics, binary_metrics, mbert_metrics, xlmr_metrics, alto_metrics):
    """Print comprehensive comparison of all 4 models"""

    print("\n" + "=" * 140)
    print("FIVE-WAY MODEL COMPARISON (5-token tolerance)")
    print("=" * 140)

    print(f"\n{'Metric':<35} {'Random':<20} {'Binary':<20} {'mBERT':<20} {'XLM-RoBERTa':<20} {'ALTO-BERT':<20}")
    print("-" * 140)

    metrics_to_show = [
        ('F-beta(2)', 'proximity_fbeta2'),
        ('Precision', 'proximity_precision'),
        ('Recall', 'proximity_recall'),
        ('F1', 'proximity_f1'),
        ('Macro F-beta(2)', 'macro_fbeta2'),
        ('Switch→Auto F-beta(2)', 'to_auto_fbeta2'),
        ('Switch→Allo F-beta(2)', 'to_allo_fbeta2'),
    ]

    for display, key in metrics_to_show:
        r = random_metrics[key]
        b = binary_metrics[key]
        m = mbert_metrics[key]
        x = xlmr_metrics[key]
        a = alto_metrics[key]

        # Mark the best (excluding random)
        best_val = max(b, m, a)
        b_mark = " ★" if b == best_val else ""
        m_mark = " ★" if m == best_val else ""
        x_mark = " ★" if x == best_val else ""
        a_mark = " ★" if a == best_val else ""

        print(f"{display:<35} {r:<20.3f} {b:<20.3f}{b_mark:<4} {m:<20.3f}{m_mark:<4} {x:<20.3f}{x_mark:<4} {a:<20.3f}{a_mark:<4}")

    print("\n" + "=" * 140)
    print("IMPROVEMENT OVER RANDOM BASELINE")
    print("=" * 140)

    key = 'proximity_fbeta2'
    baseline = random_metrics[key]

    print(f"{'Model':<30} {'F-beta(2)':<15} {'vs Random':<20}")
    print("-" * 65)
    print(f"{'Random Baseline':<30} {baseline:<15.3f} {'—':<20}")
    print(
        f"{'Binary':<30} {binary_metrics[key]:<15.3f} {f'+{(binary_metrics[key] - baseline) / baseline * 100:.1f}%':<20}")
    print(
        f"{'mBERT':<30} {mbert_metrics[key]:<15.3f} {f'+{(mbert_metrics[key] - baseline) / baseline * 100:.1f}%':<20}")
    print(
        f"{'XLM-RoBERTa':<30} {xlmr_metrics[key]:<15.3f} {f'+{(xlmr_metrics[key] - baseline) / baseline * 100:.1f}%':<20}")
    print(
        f"{'ALTO-BERT':<30} {alto_metrics[key]:<15.3f} {f'+{(alto_metrics[key] - baseline) / baseline * 100:.1f}%':<20}")

    print("\n" + "=" * 140)
    print("KEY FINDING")
    print("=" * 140)

    if alto_metrics[key] > mbert_metrics[key]:
        improvement = (alto_metrics[key] - mbert_metrics[key]) / mbert_metrics[key] * 100
        print(f"✓ Tibetan-specialized pretraining helps: ALTO-BERT outperforms mBERT by {improvement:.1f}%")
    else:
        print(f"⚠ Tibetan-specialized pretraining doesn't help: mBERT performs similarly or better")

    if alto_metrics[key] > xlmr_metrics[key]:
        improvement = (alto_metrics[key] - xlmr_metrics[key]) / xlmr_metrics[key] * 100
        print(f"✓ Tibetan-specialized pretraining helps: ALTO-BERT outperforms XLM-RoBERTa by {improvement:.1f}%")
    else:
        print(f"⚠ Tibetan-specialized pretraining doesn't help: XLM-RoBERTa performs similarly or better")

def show_detailed_segment_comparisons_five_models(test_df, binary_tokenizer, binary_session,
                                                  mbert_tokenizer, mbert_model,
                                                  xlmr_tokenizer, xlmr_model,
                                                  alto_tokenizer, alto_model,
                                                  num_examples=3, avg_switches=3.5):
    """Show detailed comparisons with all 5 models"""

    device = next(alto_model.parameters()).device

    label_names = {0: 'Auto', 1: 'Allo', 2: '→AUTO', 3: '→ALLO'}

    print("\n" + "=" * 180)
    print("DETAILED SEGMENT COMPARISON (5 MODELS)")
    print("=" * 180)

    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    for ex_num, idx in enumerate(sample_indices, 1):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()
        true_labels = [int(l) for l in row['labels'].split(',')]

        # Get predictions from all 5 models
        random_pred = process_random_model(tokens, avg_switches_per_segment=avg_switches, seed=42 + idx)
        binary_pred = process_binary_model_sentence_level(tokens, binary_tokenizer, binary_session)
        mbert_pred = process_finetuned_model(tokens, mbert_tokenizer, mbert_model, device)
        xlmr_pred = process_finetuned_model(tokens, xlmr_tokenizer, xlmr_model, device)
        alto_pred = process_finetuned_model(tokens, alto_tokenizer, alto_model, device)

        # Align lengths
        min_len = min(len(tokens), len(true_labels), len(random_pred),
                      len(binary_pred), len(mbert_pred), len(xlmr_pred), len(alto_pred))
        tokens = tokens[:min_len]
        true_labels = true_labels[:min_len]
        random_pred = random_pred[:min_len]
        binary_pred = binary_pred[:min_len]
        mbert_pred = mbert_pred[:min_len]
        xlmr_pred = xlmr_pred[:min_len]
        alto_pred = alto_pred[:min_len]

        print(f"\n{'─' * 180}")
        print(f"SEGMENT {ex_num} | File: {row['source_file'][:60]}...")
        print(f"Length: {len(tokens)} tokens | True switches: {sum(1 for l in true_labels if l in [2, 3])}")
        print(f"{'─' * 180}")

        # Calculate metrics for all models
        seg_random = evaluate_switch_detection_with_proximity(true_labels, random_pred, tolerance=5)
        seg_binary = evaluate_switch_detection_with_proximity(true_labels, binary_pred, tolerance=5)
        seg_mbert = evaluate_switch_detection_with_proximity(true_labels, mbert_pred, tolerance=5)
        seg_xlmr = evaluate_switch_detection_with_proximity(true_labels, xlmr_pred, tolerance=5)
        seg_alto = evaluate_switch_detection_with_proximity(true_labels, alto_pred, tolerance=5)

        print(f"\nSegment Metrics (F-beta(2) with 5-token tolerance):")
        print(f"  Random:   {seg_random['proximity_fbeta2']:.3f}")
        print(f"  Binary:   {seg_binary['proximity_fbeta2']:.3f}")
        print(f"  mBERT:    {seg_mbert['proximity_fbeta2']:.3f}")
        print(f"  XLM-R:    {seg_xlmr['proximity_fbeta2']:.3f}")
        print(f"  ALTO:     {seg_alto['proximity_fbeta2']:.3f}")

        # Show first 30 tokens
        print(f"\nToken-by-token comparison (first 30 tokens):")
        print(f"{'─' * 180}")
        print(f"{'Pos':<5} {'Token':<15} {'True':<8} {'Random':<8} {'Binary':<8} {'mBERT':<8} {'XLM-R':<8} {'ALTO':<8}")
        print(f"{'─' * 180}")

        for i in range(min(30, len(tokens))):
            token = tokens[i][:14]
            true_label = label_names[true_labels[i]]
            random_label = label_names[random_pred[i]]
            binary_label = label_names[binary_pred[i]]
            mbert_label = label_names[mbert_pred[i]]
            xlmr_label = label_names[xlmr_pred[i]]
            alto_label = label_names[alto_pred[i]]

            # Highlight true switches
            if true_labels[i] in [2, 3]:
                print(
                    f"[{i:3d}] {token:<15} **{true_label:<8} {random_label:<8} {binary_label:<8} {mbert_label:<8} {xlmr_label:<8} {alto_label:<8}")
            else:
                print(
                    f"[{i:3d}] {token:<15} {true_label:<8} {random_label:<8} {binary_label:<8} {mbert_label:<8} {xlmr_label:<8} {alto_label:<8}")

        # Summary counts
        print(f"\n{'─' * 180}")
        print(f"Switch Count Summary:")
        print(
            f"  True:     {sum(1 for l in true_labels if l == 2)} →AUTO, {sum(1 for l in true_labels if l == 3)} →ALLO")
        print(
            f"  Random:   {sum(1 for l in random_pred if l == 2)} →AUTO, {sum(1 for l in random_pred if l == 3)} →ALLO")
        print(
            f"  Binary:   {sum(1 for l in binary_pred if l == 2)} →AUTO, {sum(1 for l in binary_pred if l == 3)} →ALLO")
        print(f"  mBERT:    {sum(1 for l in mbert_pred if l == 2)} →AUTO, {sum(1 for l in mbert_pred if l == 3)} →ALLO")
        print(f"  XLM-R:    {sum(1 for l in xlmr_pred if l == 2)} →AUTO, {sum(1 for l in xlmr_pred if l == 3)} →ALLO")
        print(f"  ALTO:     {sum(1 for l in alto_pred if l == 2)} →AUTO, {sum(1 for l in alto_pred if l == 3)} →ALLO")


def create_detailed_segment_analysis_csv(output_file='segment_analysis_detailed.csv'):
    """
    Create detailed CSV showing per-segment performance of all models
    """
    print("\n" + "=" * 80)
    print("CREATING DETAILED SEGMENT ANALYSIS CSV")
    print("=" * 80)

    TEST_FILE = './test_segments.csv'  # UPDATE THIS

    # TEST_FILE = './dataset/annotated-data/test_segments.csv'
    test_df = pd.read_csv(TEST_FILE)

    # Calculate average switches for random model
    total_switches = 0
    for idx in range(len(test_df)):
        labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
        total_switches += sum(1 for l in labels if l in [2, 3])
    avg_switches = total_switches / len(test_df)

    # Load all models
    print("Loading all models...")
    binary_tokenizer = AutoTokenizer.from_pretrained('./alloauto-presentation/web/model')
    binary_session = ort.InferenceSession('./alloauto-presentation/web/model/onnx/model.onnx')

    mbert_tokenizer = AutoTokenizer.from_pretrained(
        './alloauto-segmentation-training/benchmark_models/mbert_baseline_model/final_model')
    mbert_model = AutoModelForTokenClassification.from_pretrained(
        './alloauto-segmentation-training/benchmark_models/mbert_baseline_model/final_model')
    mbert_model.eval()

    xlmr_tokenizer = AutoTokenizer.from_pretrained(
        './alloauto-segmentation-training/benchmark_models/xlmroberta_baseline_model/final_model')
    xlmr_model = AutoModelForTokenClassification.from_pretrained(
        './alloauto-segmentation-training/benchmark_models/xlmroberta_baseline_model/final_model')
    xlmr_model.eval()

    alto_model_id = "./tibetan_code_switching_constrained_model_wylie-final_all_data_no_labels_no_prtial_v2_2_10/final_model"
    alto_tokenizer = AutoTokenizer.from_pretrained(alto_model_id)
    alto_model = AutoModelForTokenClassification.from_pretrained(alto_model_id)
    alto_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mbert_model = mbert_model.to(device)
    xlmr_model = xlmr_model.to(device)
    alto_model = alto_model.to(device)

    print(f"Processing {len(test_df)} segments...")

    results = []

    for idx, row in test_df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing segment {idx}/{len(test_df)}...")

        segment_id = row['segment_id']
        source_file = row['source_file']
        tokens = row['tokens'].split()
        true_labels = [int(l) for l in row['labels'].split(',')]

        # Get predictions from all models
        random_pred = process_random_model(tokens, avg_switches_per_segment=avg_switches, seed=42 + idx)
        binary_pred = process_binary_model_sentence_level(tokens, binary_tokenizer, binary_session)
        mbert_pred = process_finetuned_model(tokens, mbert_tokenizer, mbert_model, device)
        xlmr_pred = process_finetuned_model(tokens, xlmr_tokenizer, xlmr_model, device)
        alto_pred = process_finetuned_model(tokens, alto_tokenizer, alto_model, device)

        # Align lengths
        min_len = min(len(true_labels), len(random_pred), len(binary_pred),
                      len(mbert_pred), len(xlmr_pred), len(alto_pred))

        true_labels = true_labels[:min_len]
        tokens = tokens[:min_len]
        random_pred = random_pred[:min_len]
        binary_pred = binary_pred[:min_len]
        mbert_pred = mbert_pred[:min_len]
        xlmr_pred = xlmr_pred[:min_len]
        alto_pred = alto_pred[:min_len]

        # Analyze each model's performance
        models = {
            'Random': random_pred,
            'Binary': binary_pred,
            'mBERT': mbert_pred,
            'XLM-R': xlmr_pred,
            'ALTO': alto_pred
        }

        # Base segment info
        segment_result = {
            'segment_id': segment_id,
            'source_file': source_file,
            'num_tokens': len(tokens),
            'tokens': ' '.join(tokens),
            'true_labels': ','.join(map(str, true_labels)),
        }

        # True switch statistics
        true_switches_to_auto = [i for i, l in enumerate(true_labels) if l == 2]
        true_switches_to_allo = [i for i, l in enumerate(true_labels) if l == 3]

        segment_result['true_total_switches'] = len(true_switches_to_auto) + len(true_switches_to_allo)
        segment_result['true_switches_to_auto'] = len(true_switches_to_auto)
        segment_result['true_switches_to_allo'] = len(true_switches_to_allo)
        segment_result['true_switch_positions'] = ','.join(map(str, true_switches_to_auto + true_switches_to_allo))

        # Analyze each model
        for model_name, pred_labels in models.items():
            # Get predictions for this model
            pred_switches_to_auto = [i for i, l in enumerate(pred_labels) if l == 2]
            pred_switches_to_allo = [i for i, l in enumerate(pred_labels) if l == 3]

            # Store predictions
            segment_result[f'{model_name}_pred_labels'] = ','.join(map(str, pred_labels))
            segment_result[f'{model_name}_pred_total_switches'] = len(pred_switches_to_auto) + len(
                pred_switches_to_allo)
            segment_result[f'{model_name}_pred_to_auto'] = len(pred_switches_to_auto)
            segment_result[f'{model_name}_pred_to_allo'] = len(pred_switches_to_allo)

            # Match true switches with predicted switches (with tolerance)
            tolerance = 5

            # Match AUTO switches
            matched_auto_exact = []
            matched_auto_proximity = []

            for true_pos in true_switches_to_auto:
                best_match = None
                best_distance = float('inf')

                for pred_pos in pred_switches_to_auto:
                    distance = abs(true_pos - pred_pos)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = pred_pos

                if best_match is not None and best_distance <= tolerance:
                    if best_distance == 0:
                        matched_auto_exact.append((true_pos, best_match, best_distance))
                    else:
                        matched_auto_proximity.append((true_pos, best_match, best_distance))

            # Match ALLO switches
            matched_allo_exact = []
            matched_allo_proximity = []

            for true_pos in true_switches_to_allo:
                best_match = None
                best_distance = float('inf')

                for pred_pos in pred_switches_to_allo:
                    distance = abs(true_pos - pred_pos)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = pred_pos

                if best_match is not None and best_distance <= tolerance:
                    if best_distance == 0:
                        matched_allo_exact.append((true_pos, best_match, best_distance))
                    else:
                        matched_allo_proximity.append((true_pos, best_match, best_distance))

            # Store match statistics
            segment_result[f'{model_name}_matched_auto_exact'] = len(matched_auto_exact)
            segment_result[f'{model_name}_matched_auto_proximity'] = len(matched_auto_proximity)
            segment_result[f'{model_name}_matched_auto_total'] = len(matched_auto_exact) + len(matched_auto_proximity)

            segment_result[f'{model_name}_matched_allo_exact'] = len(matched_allo_exact)
            segment_result[f'{model_name}_matched_allo_proximity'] = len(matched_allo_proximity)
            segment_result[f'{model_name}_matched_allo_total'] = len(matched_allo_exact) + len(matched_allo_proximity)

            # Total matches
            total_exact = len(matched_auto_exact) + len(matched_allo_exact)
            total_proximity = len(matched_auto_proximity) + len(matched_allo_proximity)

            segment_result[f'{model_name}_exact_matches'] = total_exact
            segment_result[f'{model_name}_proximity_matches'] = total_proximity
            segment_result[f'{model_name}_total_matches'] = total_exact + total_proximity

            # Missed switches
            segment_result[f'{model_name}_missed_switches'] = segment_result['true_total_switches'] - (
                        total_exact + total_proximity)

            # False positives
            total_pred = len(pred_switches_to_auto) + len(pred_switches_to_allo)
            segment_result[f'{model_name}_false_positives'] = total_pred - (total_exact + total_proximity)

            # Store detailed match info
            auto_match_details = '; '.join(
                [f"true:{t}→pred:{p}(dist:{d})" for t, p, d in matched_auto_exact + matched_auto_proximity])
            allo_match_details = '; '.join(
                [f"true:{t}→pred:{p}(dist:{d})" for t, p, d in matched_allo_exact + matched_allo_proximity])

            segment_result[f'{model_name}_auto_match_details'] = auto_match_details if auto_match_details else 'none'
            segment_result[f'{model_name}_allo_match_details'] = allo_match_details if allo_match_details else 'none'

            # Calculate metrics for this segment
            true_total = segment_result['true_total_switches']
            pred_total = total_pred
            matched_total = total_exact + total_proximity

            precision = matched_total / pred_total if pred_total > 0 else 0
            recall = matched_total / true_total if true_total > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            segment_result[f'{model_name}_precision'] = round(precision, 3)
            segment_result[f'{model_name}_recall'] = round(recall, 3)
            segment_result[f'{model_name}_f1'] = round(f1, 3)

        results.append(segment_result)

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Save to CSV
    df_results.to_csv(output_file, index=False)

    print(f"\n✅ Detailed analysis saved to: {output_file}")
    print(f"   Total segments: {len(df_results)}")
    print(f"   Total columns: {len(df_results.columns)}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for model_name in ['Random', 'Binary', 'mBERT', 'XLM-R', 'ALTO']:
        print(f"\n{model_name}:")
        print(f"  Avg Exact Matches: {df_results[f'{model_name}_exact_matches'].mean():.2f}")
        print(f"  Avg Proximity Matches: {df_results[f'{model_name}_proximity_matches'].mean():.2f}")
        print(f"  Avg Total Matches: {df_results[f'{model_name}_total_matches'].mean():.2f}")
        print(f"  Avg Precision: {df_results[f'{model_name}_precision'].mean():.3f}")
        print(f"  Avg Recall: {df_results[f'{model_name}_recall'].mean():.3f}")
        print(f"  Avg F1: {df_results[f'{model_name}_f1'].mean():.3f}")

    return df_results


def process_crf_model(tokens, tokenizer, model, device):
    """Process tokens through CRF-enhanced model (uses Viterbi decoding)"""
    tokenizer_output = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Filter inputs - only pass what the model expects
    inputs = {
        'input_ids': tokenizer_output['input_ids'].to(device),
        'attention_mask': tokenizer_output['attention_mask'].to(device)
    }

    with torch.no_grad():
        outputs = model(**inputs)
        # CRF returns predictions from Viterbi
        viterbi_predictions = outputs['predictions'][0]

    # Align to original tokens
    word_ids = tokenizer_output.word_ids()
    aligned_preds = []
    previous_word_idx = None
    pred_idx = 0

    for word_idx in word_ids:
        if word_idx is not None and word_idx != previous_word_idx:
            if pred_idx < len(viterbi_predictions):
                aligned_preds.append(viterbi_predictions[pred_idx])
                pred_idx += 1
        previous_word_idx = word_idx

    return aligned_preds

def print_fbeta_comparison_six_models(random_metrics, binary_metrics, mbert_metrics, xlmr_metrics, alto_metrics,
                                      crf_metrics):
    """Print comprehensive comparison of all 6 models"""

    print("\n" + "=" * 160)
    print("SIX-WAY MODEL COMPARISON (5-token tolerance)")
    print("=" * 160)

    print(f"\n{'Metric':<35} {'Random':<18} {'Binary':<18} {'mBERT':<18} {'XLM-R':<18} {'ALTO':<18} {'CRF':<18}")
    print("-" * 160)

    metrics_to_show = [
        ('F-beta(2)', 'proximity_fbeta2'),
        ('Precision', 'proximity_precision'),
        ('Recall', 'proximity_recall'),
        ('F1', 'proximity_f1'),
        ('Macro F-beta(2)', 'macro_fbeta2'),
        ('Switch→Auto F-beta(2)', 'to_auto_fbeta2'),
        ('Switch→Allo F-beta(2)', 'to_allo_fbeta2'),
    ]

    for display, key in metrics_to_show:
        r = random_metrics[key]
        b = binary_metrics[key]
        m = mbert_metrics[key]
        x = xlmr_metrics[key]
        a = alto_metrics[key]
        c = crf_metrics[key]

        # Mark the best (excluding random)
        best_val = max(b, m, x, a, c)
        b_mark = " ★" if b == best_val else ""
        m_mark = " ★" if m == best_val else ""
        x_mark = " ★" if x == best_val else ""
        a_mark = " ★" if a == best_val else ""
        c_mark = " ★" if c == best_val else ""

        print(f"{display:<35} {r:<18.3f} {b:<15.3f}{b_mark:<3} {m:<15.3f}{m_mark:<3} "
              f"{x:<15.3f}{x_mark:<3} {a:<15.3f}{a_mark:<3} {c:<15.3f}{c_mark:<3}")

    print("\n" + "=" * 160)
    print("KEY FINDINGS")
    print("=" * 160)

    key = 'proximity_fbeta2'

    print(f"\nCRF vs ALTO Comparison:")
    if crf_metrics[key] > alto_metrics[key]:
        improvement = (crf_metrics[key] - alto_metrics[key]) / alto_metrics[key] * 100
        print(f"  ✓ CRF improves over ALTO by {improvement:.1f}%")
        print(f"  CRF: {crf_metrics[key]:.3f} vs ALTO: {alto_metrics[key]:.3f}")
    else:
        decline = (alto_metrics[key] - crf_metrics[key]) / alto_metrics[key] * 100
        print(f"  ⚠ CRF performs {decline:.1f}% worse than ALTO")
        print(f"  ALTO: {alto_metrics[key]:.3f} vs CRF: {crf_metrics[key]:.3f}")

    print(f"\nBest overall model: ", end="")
    best_model = max([('Binary', b), ('mBERT', m), ('XLM-R', x), ('ALTO', a), ('CRF', c)],
                     key=lambda x: binary_metrics[key] if x[0] == 'Binary' else
                     mbert_metrics[key] if x[0] == 'mBERT' else
                     xlmr_metrics[key] if x[0] == 'XLM-R' else
                     alto_metrics[key] if x[0] == 'ALTO' else
                     crf_metrics[key])
    print(f"{best_model[0]} (F-beta(2): {best_model[1]:.3f})")


def analyze_performance_by_segment_type(test_df, all_true_labels, model_predictions_dict, tolerance=5):
    """
    Analyze model performance separately for segments WITH and WITHOUT switches

    Args:
        test_df: Test dataframe with 'has_switch' column
        all_true_labels: All true labels (flattened) - MUST match order of test_df
        model_predictions_dict: Dict of {model_name: predictions_list}
        tolerance: Proximity tolerance for matching

    Returns:
        Dictionary with metrics by segment type
    """
    print("\n" + "=" * 100)
    print("PERFORMANCE BY SEGMENT TYPE (WITH/WITHOUT SWITCHES)")
    print("=" * 100)

    # Check if has_switch column exists
    if 'has_switch' not in test_df.columns:
        print("⚠️ 'has_switch' column not found in test data. Computing from labels...")
        test_df['has_switch'] = test_df['labels'].apply(
            lambda x: any(int(l) in [2, 3] for l in x.split(','))
        )

    # Count segments by type
    segments_with_switches = test_df[test_df['has_switch'] == True]
    segments_without_switches = test_df[test_df['has_switch'] == False]

    print(f"\nSegment distribution:")
    print(
        f"  WITH switches: {len(segments_with_switches)} segments ({len(segments_with_switches) / len(test_df) * 100:.1f}%)")
    print(
        f"  WITHOUT switches: {len(segments_without_switches)} segments ({len(segments_without_switches) / len(test_df) * 100:.1f}%)")

    # Build indices mapping - track where each segment's tokens start/end in flattened arrays
    segment_boundaries = []
    current_position = 0

    for idx, row in test_df.iterrows():
        num_tokens = len(row['labels'].split(','))
        segment_boundaries.append({
            'start': current_position,
            'end': current_position + num_tokens,
            'has_switch': row['has_switch'],
            'num_tokens': num_tokens
        })
        current_position += num_tokens

    # Extract labels and predictions by segment type
    def extract_by_segment_type(flat_array, has_switch_value):
        """Extract elements from flattened array for segments of given type"""
        result = []
        for i, boundary in enumerate(segment_boundaries):
            if boundary['has_switch'] == has_switch_value:
                result.extend(flat_array[boundary['start']:boundary['end']])
        return result

    # Get true labels by segment type
    true_with_switches = extract_by_segment_type(all_true_labels, True)
    true_without_switches = extract_by_segment_type(all_true_labels, False)

    print(f"\nToken distribution:")
    print(f"  Tokens in segments WITH switches: {len(true_with_switches)}")
    print(f"  Tokens in segments WITHOUT switches: {len(true_without_switches)}")

    # Verify alignment
    total_extracted = len(true_with_switches) + len(true_without_switches)
    if total_extracted != len(all_true_labels):
        print(f"⚠️ WARNING: Alignment issue detected!")
        print(f"  Expected: {len(all_true_labels)} tokens")
        print(f"  Got: {total_extracted} tokens")

    # Calculate metrics for each model on both segment types
    results = {}

    for model_name, all_preds in model_predictions_dict.items():
        # Verify prediction length matches
        if len(all_preds) != len(all_true_labels):
            print(f"⚠️ WARNING: {model_name} predictions length mismatch!")
            print(f"  True labels: {len(all_true_labels)}, Predictions: {len(all_preds)}")
            # Truncate to match
            all_preds = all_preds[:len(all_true_labels)]

        # Get predictions by segment type
        preds_with_switches = extract_by_segment_type(all_preds, True)
        preds_without_switches = extract_by_segment_type(all_preds, False)

        # Verify lengths match
        if len(preds_with_switches) != len(true_with_switches):
            print(
                f"⚠️ Length mismatch for {model_name} (with switches): {len(preds_with_switches)} vs {len(true_with_switches)}")
            min_len = min(len(preds_with_switches), len(true_with_switches))
            preds_with_switches = preds_with_switches[:min_len]
            true_with_switches_eval = true_with_switches[:min_len]
        else:
            true_with_switches_eval = true_with_switches

        if len(preds_without_switches) != len(true_without_switches):
            print(
                f"⚠️ Length mismatch for {model_name} (without switches): {len(preds_without_switches)} vs {len(true_without_switches)}")
            min_len = min(len(preds_without_switches), len(true_without_switches))
            preds_without_switches = preds_without_switches[:min_len]
            true_without_switches_eval = true_without_switches[:min_len]
        else:
            true_without_switches_eval = true_without_switches

        # Evaluate on segments WITH switches
        if len(true_with_switches_eval) > 0:
            metrics_with = evaluate_switch_detection_with_proximity(
                true_with_switches_eval, preds_with_switches, tolerance
            )
        else:
            metrics_with = None

        # Evaluate on segments WITHOUT switches
        if len(true_without_switches_eval) > 0:
            true_arr = np.array(true_without_switches_eval)
            pred_arr = np.array(preds_without_switches)

            # Count false switches (model predicts switch when there shouldn't be any)
            false_switches = np.sum((pred_arr == 2) | (pred_arr == 3))
            total_tokens = len(true_without_switches_eval)

            # Mode accuracy (0,2 -> auto mode, 1,3 -> allo mode)
            true_modes = (true_arr % 2)  # 0,2->0 (auto), 1,3->1 (allo)
            pred_modes = (pred_arr % 2)

            mode_accuracy = (true_modes == pred_modes).mean()
            false_switch_rate = false_switches / total_tokens if total_tokens > 0 else 0

            metrics_without = {
                'total_tokens': total_tokens,
                'false_switches': int(false_switches),
                'false_switch_rate': false_switch_rate,
                'mode_accuracy': mode_accuracy
            }
        else:
            metrics_without = None

        results[model_name] = {
            'with_switches': metrics_with,
            'without_switches': metrics_without
        }

    # Print results
    print("\n" + "=" * 100)
    print("SEGMENTS WITH SWITCHES (Switch Detection Performance)")
    print("=" * 100)
    print(f"{'Model':<15} {'F-beta(2)':<12} {'Precision':<12} {'Recall':<12} {'True SW':<10} {'Pred SW':<10}")
    print("-" * 100)

    for model_name in model_predictions_dict.keys():
        metrics = results[model_name]['with_switches']
        if metrics:
            print(f"{model_name:<15} "
                  f"{metrics['proximity_fbeta2']:<12.3f} "
                  f"{metrics['proximity_precision']:<12.3f} "
                  f"{metrics['proximity_recall']:<12.3f} "
                  f"{metrics['true_switches']:<10} "
                  f"{metrics['pred_switches']:<10}")

    print("\n" + "=" * 100)
    print("SEGMENTS WITHOUT SWITCHES (False Positive Control)")
    print("=" * 100)
    print(f"{'Model':<15} {'Mode Accuracy':<15} {'False Switches':<15} {'False SW Rate':<15} {'Total Tokens':<15}")
    print("-" * 100)

    for model_name in model_predictions_dict.keys():
        metrics = results[model_name]['without_switches']
        if metrics:
            print(f"{model_name:<15} "
                  f"{metrics['mode_accuracy']:<15.3f} "
                  f"{metrics['false_switches']:<15} "
                  f"{metrics['false_switch_rate']:<15.4f} "
                  f"{metrics['total_tokens']:<15}")

    # Summary insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    # Find best models
    with_switch_models = [(name, results[name]['with_switches'])
                          for name in model_predictions_dict.keys()
                          if results[name]['with_switches']]

    without_switch_models = [(name, results[name]['without_switches'])
                             for name in model_predictions_dict.keys()
                             if results[name]['without_switches']]

    if with_switch_models:
        best_fbeta_model = max(with_switch_models, key=lambda x: x[1]['proximity_fbeta2'])
        print(
            f"\nBest at detecting switches: {best_fbeta_model[0]} (F-beta(2): {best_fbeta_model[1]['proximity_fbeta2']:.3f})")

    if without_switch_models:
        best_mode_acc_model = max(without_switch_models, key=lambda x: x[1]['mode_accuracy'])
        lowest_false_sw_model = min(without_switch_models, key=lambda x: x[1]['false_switch_rate'])

        print(
            f"Best mode accuracy (non-switching): {best_mode_acc_model[0]} ({best_mode_acc_model[1]['mode_accuracy']:.3f})")
        print(
            f"Lowest false switch rate: {lowest_false_sw_model[0]} ({lowest_false_sw_model[1]['false_switch_rate']:.4f})")

        # Check if any model has problematically high false switch rate
        for name, metrics in without_switch_models:
            if metrics['false_switch_rate'] > 0.05:  # More than 5% false switches
                print(f"  ⚠️ {name} has high false switch rate: {metrics['false_switch_rate']:.1%}")

    return results
def analyze_performance_by_segment_type_old(test_df, all_true_labels, model_predictions_dict, tolerance=5):
    """
    Analyze model performance separately for segments WITH and WITHOUT switches

    Args:
        test_df: Test dataframe with 'has_switch' column
        all_true_labels: All true labels (flattened)
        model_predictions_dict: Dict of {model_name: predictions_list}
        tolerance: Proximity tolerance for matching

    Returns:
        Dictionary with metrics by segment type
    """
    print("\n" + "=" * 100)
    print("PERFORMANCE BY SEGMENT TYPE (WITH/WITHOUT SWITCHES)")
    print("=" * 100)

    # Check if has_switch column exists
    if 'has_switch' not in test_df.columns:
        print("⚠️ 'has_switch' column not found in test data. Computing from labels...")
        test_df['has_switch'] = test_df['labels'].apply(
            lambda x: any(int(l) in [2, 3] for l in x.split(','))
        )

    # Split segments by type
    segments_with_switches = test_df[test_df['has_switch'] == True].index.tolist()
    segments_without_switches = test_df[test_df['has_switch'] == False].index.tolist()

    print(f"\nSegment distribution:")
    print(
        f"  WITH switches: {len(segments_with_switches)} segments ({len(segments_with_switches) / len(test_df) * 100:.1f}%)")
    print(
        f"  WITHOUT switches: {len(segments_without_switches)} segments ({len(segments_without_switches) / len(test_df) * 100:.1f}%)")

    # Collect labels by segment type
    def get_labels_for_segments(segment_indices):
        """Extract labels for specific segments"""
        labels_list = []
        current_position = 0

        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            segment_labels = [int(l) for l in row['labels'].split(',')]

            if idx in segment_indices:
                labels_list.extend(segment_labels)

            current_position += len(segment_labels)

        return labels_list

    # Get predictions by segment type
    def get_predictions_for_segments(all_predictions, segment_indices):
        """Extract predictions for specific segments"""
        preds_list = []
        current_position = 0

        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            num_tokens = len(row['labels'].split(','))

            if idx in segment_indices:
                segment_preds = all_predictions[current_position:current_position + num_tokens]
                preds_list.extend(segment_preds)

            current_position += num_tokens

        return preds_list

    # Get true labels by segment type
    true_with_switches = get_labels_for_segments(segments_with_switches)
    true_without_switches = get_labels_for_segments(segments_without_switches)

    print(f"\nToken distribution:")
    print(f"  Tokens in segments WITH switches: {len(true_with_switches)}")
    print(f"  Tokens in segments WITHOUT switches: {len(true_without_switches)}")

    # Calculate metrics for each model on both segment types
    results = {}

    for model_name, all_preds in model_predictions_dict.items():
        # Get predictions by segment type
        preds_with_switches = get_predictions_for_segments(all_preds, segments_with_switches)
        preds_without_switches = get_predictions_for_segments(all_preds, segments_without_switches)

        # Evaluate on segments WITH switches
        if len(true_with_switches) > 0:
            metrics_with = evaluate_switch_detection_with_proximity(
                true_with_switches, preds_with_switches, tolerance
            )
        else:
            metrics_with = None

        # Evaluate on segments WITHOUT switches (different metrics needed)
        if len(true_without_switches) > 0:
            # For non-switching segments, we care about:
            # 1. Not predicting false switches
            # 2. Correct mode (auto vs allo)

            true_arr = np.array(true_without_switches)
            pred_arr = np.array(preds_without_switches)

            # Count false switches (model predicts switch when there shouldn't be any)
            false_switches = np.sum((pred_arr == 2) | (pred_arr == 3))
            total_tokens = len(true_without_switches)

            # Mode accuracy (ignoring switch labels)
            # Map: 0,2 -> 0 (auto), 1,3 -> 1 (allo)
            true_modes = (true_arr >= 1).astype(int)  # 0/2->0 (auto), 1/3->1 (allo)
            pred_modes = (pred_arr >= 1).astype(int)

            mode_accuracy = (true_modes == pred_modes).mean()
            false_switch_rate = false_switches / total_tokens if total_tokens > 0 else 0

            metrics_without = {
                'total_tokens': total_tokens,
                'false_switches': false_switches,
                'false_switch_rate': false_switch_rate,
                'mode_accuracy': mode_accuracy
            }
        else:
            metrics_without = None

        results[model_name] = {
            'with_switches': metrics_with,
            'without_switches': metrics_without
        }

    # Print results
    print("\n" + "=" * 100)
    print("SEGMENTS WITH SWITCHES")
    print("=" * 100)
    print(f"{'Model':<15} {'F-beta(2)':<12} {'Precision':<12} {'Recall':<12} {'True SW':<10} {'Pred SW':<10}")
    print("-" * 100)

    for model_name in model_predictions_dict.keys():
        metrics = results[model_name]['with_switches']
        if metrics:
            print(f"{model_name:<15} "
                  f"{metrics['proximity_fbeta2']:<12.3f} "
                  f"{metrics['proximity_precision']:<12.3f} "
                  f"{metrics['proximity_recall']:<12.3f} "
                  f"{metrics['true_switches']:<10} "
                  f"{metrics['pred_switches']:<10}")

    print("\n" + "=" * 100)
    print("SEGMENTS WITHOUT SWITCHES")
    print("=" * 100)
    print(f"{'Model':<15} {'Mode Accuracy':<15} {'False Switches':<15} {'False SW Rate':<15}")
    print("-" * 100)

    for model_name in model_predictions_dict.keys():
        metrics = results[model_name]['without_switches']
        if metrics:
            print(f"{model_name:<15} "
                  f"{metrics['mode_accuracy']:<15.3f} "
                  f"{metrics['false_switches']:<15} "
                  f"{metrics['false_switch_rate']:<15.3f}")

    # Summary insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    # Find best model for each category
    best_fbeta_model = max(
        [(name, results[name]['with_switches']['proximity_fbeta2'])
         for name in model_predictions_dict.keys() if results[name]['with_switches']],
        key=lambda x: x[1]
    )

    best_mode_acc_model = max(
        [(name, results[name]['without_switches']['mode_accuracy'])
         for name in model_predictions_dict.keys() if results[name]['without_switches']],
        key=lambda x: x[1]
    )

    lowest_false_sw_model = min(
        [(name, results[name]['without_switches']['false_switch_rate'])
         for name in model_predictions_dict.keys() if results[name]['without_switches']],
        key=lambda x: x[1]
    )

    print(f"\nBest at detecting switches: {best_fbeta_model[0]} (F-beta(2): {best_fbeta_model[1]:.3f})")
    print(f"Best mode accuracy (non-switching): {best_mode_acc_model[0]} ({best_mode_acc_model[1]:.3f})")
    print(f"Lowest false switch rate: {lowest_false_sw_model[0]} ({lowest_false_sw_model[1]:.3f})")

    return results


def print_six_model_comparison_by_segment_type(segment_type_results, model_names):
    """
    Print comprehensive comparison showing WITH/WITHOUT switch performance side-by-side

    Args:
        segment_type_results: Results from analyze_performance_by_segment_type()
        model_names: List of model names in order (e.g., ['Random', 'Binary', 'mBERT', 'XLM-R', 'ALTO', 'CRF'])
    """
    print("\n" + "=" * 180)
    print("SIX-WAY MODEL COMPARISON BY SEGMENT TYPE (5-token tolerance)")
    print("=" * 180)

    # Part 1: Segments WITH switches
    print("\n" + "─" * 180)
    print("SEGMENTS WITH SWITCHES (Switch Detection Performance)")
    print("─" * 180)
    print(
        f"{'Model':<15} {'F-beta(2)':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'True SW':<10} {'Pred SW':<10} {'Matched':<10}")
    print("-" * 180)

    with_switch_scores = []
    for model_name in model_names:
        metrics = segment_type_results[model_name]['with_switches']
        if metrics:
            print(f"{model_name:<15} "
                  f"{metrics['proximity_fbeta2']:<12.3f} "
                  f"{metrics['proximity_precision']:<12.3f} "
                  f"{metrics['proximity_recall']:<12.3f} "
                  f"{metrics['proximity_f1']:<12.3f} "
                  f"{metrics['true_switches']:<10} "
                  f"{metrics['pred_switches']:<10} "
                  f"{metrics['total_matches']:<10}")
            with_switch_scores.append((model_name, metrics['proximity_fbeta2']))
        else:
            print(f"{model_name:<15} {'N/A':<12}")

    # Mark the best
    if with_switch_scores:
        best_model, best_score = max(with_switch_scores, key=lambda x: x[1])
        print(f"\n★ Best performer: {best_model} (F-beta(2): {best_score:.3f})")

    # Part 2: Segments WITHOUT switches
    print("\n" + "─" * 180)
    print("SEGMENTS WITHOUT SWITCHES (False Positive Control & Mode Accuracy)")
    print("─" * 180)
    print(
        f"{'Model':<15} {'Mode Acc':<12} {'False SW':<12} {'False SW Rate':<15} {'Total Tokens':<15} {'Interpretation'}")
    print("-" * 180)

    mode_acc_scores = []
    false_sw_scores = []

    for model_name in model_names:
        metrics = segment_type_results[model_name]['without_switches']
        if metrics:
            # Interpretation
            if metrics['false_switch_rate'] < 0.01:
                interp = "Excellent control"
            elif metrics['false_switch_rate'] < 0.03:
                interp = "Good control"
            elif metrics['false_switch_rate'] < 0.05:
                interp = "Fair control"
            else:
                interp = "⚠️ High false positives"

            print(f"{model_name:<15} "
                  f"{metrics['mode_accuracy']:<12.3f} "
                  f"{metrics['false_switches']:<12} "
                  f"{metrics['false_switch_rate']:<15.4f} "
                  f"{metrics['total_tokens']:<15} "
                  f"{interp}")

            mode_acc_scores.append((model_name, metrics['mode_accuracy']))
            false_sw_scores.append((model_name, metrics['false_switch_rate']))
        else:
            print(f"{model_name:<15} {'N/A':<12}")

    # Mark the best
    if mode_acc_scores:
        best_mode_model, best_mode_acc = max(mode_acc_scores, key=lambda x: x[1])
        best_control_model, best_control = min(false_sw_scores, key=lambda x: x[1])

        print(f"\n★ Best mode accuracy: {best_mode_model} ({best_mode_acc:.3f})")
        print(f"★ Best false positive control: {best_control_model} (rate: {best_control:.4f})")

    # Part 3: Overall assessment
    print("\n" + "=" * 180)
    print("OVERALL ASSESSMENT")
    print("=" * 180)

    # Calculate a composite score: high switch detection + low false positives + high mode accuracy
    print(f"\n{'Model':<15} {'Switch F-beta(2)':<18} {'Mode Accuracy':<18} {'False SW Rate':<18} {'Overall Quality'}")
    print("-" * 180)

    for model_name in model_names:
        with_metrics = segment_type_results[model_name]['with_switches']
        without_metrics = segment_type_results[model_name]['without_switches']

        if with_metrics and without_metrics:
            # Composite quality score
            switch_score = with_metrics['proximity_fbeta2']
            mode_score = without_metrics['mode_accuracy']
            fp_penalty = without_metrics['false_switch_rate']

            # Quality rating: good at switches + accurate on modes - false positives
            quality = (switch_score + mode_score) / 2 - fp_penalty * 2

            quality_label = ""
            if quality >= 0.85:
                quality_label = "★★★ Excellent"
            elif quality >= 0.75:
                quality_label = "★★ Very Good"
            elif quality >= 0.65:
                quality_label = "★ Good"
            else:
                quality_label = "Needs improvement"

            print(f"{model_name:<15} "
                  f"{switch_score:<18.3f} "
                  f"{mode_score:<18.3f} "
                  f"{fp_penalty:<18.4f} "
                  f"{quality_label}")

    # Key insights
    print("\n" + "=" * 180)
    print("KEY INSIGHTS")
    print("=" * 180)

    # Compare models
    if len(model_names) >= 2:
        # Find models that are good at switches but have high FP
        for model_name in model_names:
            with_m = segment_type_results[model_name]['with_switches']
            without_m = segment_type_results[model_name]['without_switches']

            if with_m and without_m:
                if with_m['proximity_fbeta2'] > 0.7 and without_m['false_switch_rate'] > 0.05:
                    print(f"\n⚠️ {model_name}: Good at finding switches but produces many false positives")
                elif with_m['proximity_fbeta2'] < 0.5 and without_m['false_switch_rate'] < 0.02:
                    print(f"\n⚠️ {model_name}: Conservative (few false positives) but misses many real switches")
                elif with_m['proximity_fbeta2'] > 0.7 and without_m['false_switch_rate'] < 0.02:
                    print(f"\n✓ {model_name}: Well-balanced (good detection + low false positives)")

    print("\n" + "=" * 180)
# Update unified_evaluation() to call this analysis
if __name__ == "__main__":
    # Returns 5 models now
    random_metrics, binary_metrics, mbert_metrics, xlmr_metrics, alto_metrics, crf_metrics, segment_type_results = unified_evaluation()
    print_six_model_comparison_by_segment_type(
        segment_type_results,
        ['Random', 'Binary', 'mBERT', 'XLM-R', 'ALTO', 'CRF']
    )

    if random_metrics is not None:
        print("\n" + "=" * 160)
        print("DETAILED 5-MODEL COMPARISON")
        print("=" * 160)

        # Compare all FIVE models

        print_fbeta_comparison_six_models(random_metrics, binary_metrics, mbert_metrics,
                                          xlmr_metrics, alto_metrics, crf_metrics)
        # Show detailed segment comparisons
        print("\n\nLoading models for detailed segment comparisons...")
        TEST_FILE = './test_segments.csv'
        # TEST_FILE = './dataset/annotated-data/test_segments.csv'
        test_df = pd.read_csv(TEST_FILE)

        # Calculate average switches
        total_switches = 0
        for idx in range(len(test_df)):
            labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
            total_switches += sum(1 for l in labels if l in [2, 3])
        avg_switches = total_switches / len(test_df)

        # Load all models
        binary_tokenizer = AutoTokenizer.from_pretrained('./alloauto-presentation/web/model')
        binary_session = ort.InferenceSession('./alloauto-presentation/web/model/onnx/model.onnx')

        mbert_tokenizer = AutoTokenizer.from_pretrained('./alloauto-segmentation-training/benchmark_models/mbert_baseline_model/final_model')
        mbert_model = AutoModelForTokenClassification.from_pretrained('./alloauto-segmentation-training/benchmark_models/mbert_baseline_model/final_model')
        mbert_model.eval()

        xlmr_tokenizer = AutoTokenizer.from_pretrained('./alloauto-segmentation-training/benchmark_models/xlmroberta_baseline_model/final_model')
        xlmr_model = AutoModelForTokenClassification.from_pretrained('./alloauto-segmentation-training/benchmark_models/xlmroberta_baseline_model/final_model')
        xlmr_model.eval()

        alto_model_id = "./tibetan_code_switching_constrained_model_wylie-final_all_data_no_labels_no_prtial_v2_2_10/final_model"
        alto_tokenizer = AutoTokenizer.from_pretrained(alto_model_id)
        alto_model = AutoModelForTokenClassification.from_pretrained(alto_model_id)
        alto_model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mbert_model = mbert_model.to(device)
        xlmr_model = xlmr_model.to(device)
        alto_model = alto_model.to(device)

        # Show detailed comparisons
        show_detailed_segment_comparisons_five_models(
            test_df, binary_tokenizer, binary_session,
            mbert_tokenizer, mbert_model,
            xlmr_tokenizer, xlmr_model,
            alto_tokenizer, alto_model,
            num_examples=3, avg_switches=avg_switches
        )

        print("\n\n" + "=" * 80)
        print("CREATING DETAILED SEGMENT ANALYSIS")
        print("=" * 80)

        df_detailed = create_detailed_segment_analysis_csv('segment_analysis_detailed.csv')

        # Optional: Show some examples
        print("\n" + "=" * 80)
        print("EXAMPLE: First 3 segments")
        print("=" * 80)

        for idx in range(min(3, len(df_detailed))):
            row = df_detailed.iloc[idx]
            print(f"\nSegment {idx + 1}: {row['segment_id']}")
            print(f"  File: {row['source_file']}")
            print(
                f"  True switches: {row['true_total_switches']} (→AUTO: {row['true_switches_to_auto']}, →ALLO: {row['true_switches_to_allo']})")
            print(f"  Model performance:")
            for model in ['Random', 'Binary', 'mBERT', 'XLM-R', 'ALTO']:
                print(
                    f"    {model:10s}: Exact={row[f'{model}_exact_matches']}, Proximity={row[f'{model}_proximity_matches']}, F1={row[f'{model}_f1']:.3f}")

        # import ipdb
        # ipdb.set_trace()