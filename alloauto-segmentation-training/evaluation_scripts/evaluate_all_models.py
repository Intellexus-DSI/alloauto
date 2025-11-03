"""
Comprehensive Code-Switching Evaluation Framework
Evaluates models on ALL segments, switch segments, and non-switch segments
Includes fixed post-processing for mBERT, XLM-R, and ALTO
"""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import re
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
import os

# Import your existing modules (adjust paths as needed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
### POST PROCESS func

"""
Post-processing module for code-switching predictions
Enforces logical consistency rules for switch points
"""

import numpy as np
from typing import Union, List


def apply_post_processing_rules(predictions: Union[np.ndarray, List[int]],
                                min_tokens_between_switches: int = 2) -> Union[np.ndarray, List[int]]:
    """
    Apply post-processing rules to enforce logical consistency in predictions.

    Rules enforced:
    1. No adjacent switches (switches cannot be back-to-back)
    2. Switch→Allo (label 3) must follow Non-switch Auto (label 0)
    3. Switch→Auto (label 2) must follow Non-switch Allo (label 1)
    4. Minimum tokens between switches (default: 2)
       - After Switch→Allo: at least 2 Allo tokens before next switch
       - After Switch→Auto: at least 2 Auto tokens before next switch

    Label mapping:
        0: Non-switch Auto
        1: Non-switch Allo
        2: Switch→Auto
        3: Switch→Allo

    Args:
        predictions: Array or list of predicted labels
        min_tokens_between_switches: Minimum number of tokens required between switches

    Returns:
        Corrected predictions in same format as input
    """
    if len(predictions) == 0:
        return predictions

    # Convert to list for easier manipulation
    is_numpy = isinstance(predictions, np.ndarray)
    preds = predictions.tolist() if is_numpy else list(predictions)

    corrected = []
    current_mode = 0  # Start in Auto mode (0)
    tokens_since_last_switch = min_tokens_between_switches

    for i, pred in enumerate(preds):
        # Skip padding tokens (used in batched processing)
        if pred == -100:
            corrected.append(pred)
            continue

        # Identify if current prediction is a switch
        is_switch = pred in [2, 3]

        if is_switch:
            # RULE 1: No adjacent switches
            if tokens_since_last_switch == 0:
                # Previous token was a switch - convert to continuation
                corrected_label = current_mode  # Continue in current mode
                corrected.append(corrected_label)
                tokens_since_last_switch += 1
                continue

            # RULE 4: Minimum tokens between switches
            if tokens_since_last_switch < min_tokens_between_switches:
                # Too soon to switch - need more tokens in current mode
                corrected_label = current_mode  # Continue in current mode
                corrected.append(corrected_label)
                tokens_since_last_switch += 1
                continue

            # RULE 2 & 3: Mode consistency check
            if pred == 2:  # Switch→Auto
                if current_mode != 1:
                    # INVALID: Can only switch to Auto from Allo mode
                    # Convert to continuation in current mode
                    corrected.append(current_mode)
                    tokens_since_last_switch += 1
                else:
                    # VALID: Switching from Allo to Auto
                    corrected.append(2)
                    current_mode = 0  # Now in Auto mode
                    tokens_since_last_switch = 0  # Reset counter

            elif pred == 3:  # Switch→Allo
                if current_mode != 0:
                    # INVALID: Can only switch to Allo from Auto mode
                    # Convert to continuation in current mode
                    corrected.append(current_mode)
                    tokens_since_last_switch += 1
                else:
                    # VALID: Switching from Auto to Allo
                    corrected.append(3)
                    current_mode = 1  # Now in Allo mode
                    tokens_since_last_switch = 0  # Reset counter
        else:
            # Non-switch token (0 or 1)
            # Update current mode based on prediction
            if pred == 0:
                current_mode = 0  # Auto mode
            elif pred == 1:
                current_mode = 1  # Allo mode

            corrected.append(pred)
            tokens_since_last_switch += 1

    # Return in same format as input
    return np.array(corrected) if is_numpy else corrected


def verify_post_processing_rules(predictions: Union[np.ndarray, List[int]]) -> dict:
    """
    Verify that predictions follow all post-processing rules.
    Useful for debugging and validation.

    Args:
        predictions: Array or list of predicted labels

    Returns:
        Dictionary with violation counts and details
    """
    preds = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions

    violations = {
        'adjacent_switches': [],
        'invalid_mode_transitions': [],
        'insufficient_spacing': [],
        'total_switches': 0
    }

    current_mode = 0
    tokens_since_last_switch = 0
    min_spacing = 2

    for i, pred in enumerate(preds):
        if pred == -100:
            continue

        is_switch = pred in [2, 3]

        if is_switch:
            violations['total_switches'] += 1

            # Check adjacent switches
            if tokens_since_last_switch == 0:
                violations['adjacent_switches'].append(i)

            # Check spacing
            if tokens_since_last_switch < min_spacing:
                violations['insufficient_spacing'].append({
                    'position': i,
                    'tokens_since_last': tokens_since_last_switch
                })

            # Check mode consistency
            if pred == 2 and current_mode != 1:
                violations['invalid_mode_transitions'].append({
                    'position': i,
                    'switch_type': '→Auto',
                    'current_mode': 'Auto' if current_mode == 0 else 'Allo'
                })
            elif pred == 3 and current_mode != 0:
                violations['invalid_mode_transitions'].append({
                    'position': i,
                    'switch_type': '→Allo',
                    'current_mode': 'Auto' if current_mode == 0 else 'Allo'
                })

            # Update state
            if pred == 2:
                current_mode = 0
            elif pred == 3:
                current_mode = 1
            tokens_since_last_switch = 0
        else:
            # Update mode
            if pred == 0:
                current_mode = 0
            elif pred == 1:
                current_mode = 1
            tokens_since_last_switch += 1

    return violations


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def process_model_predictions_with_post_processing(tokens: List[str],
                                                   tokenizer,
                                                   model,
                                                   device,
                                                   min_tokens_between_switches: int = 2) -> List[int]:
    """
    Complete pipeline: Get model predictions and apply post-processing.
    Works with any fine-tuned model (ALTO, mBERT, XLM-R, etc.)

    Args:
        tokens: List of input tokens
        tokenizer: Model tokenizer
        model: Fine-tuned model
        device: torch device
        min_tokens_between_switches: Minimum spacing between switches

    Returns:
        Post-processed predictions
    """
    import torch

    # Tokenize
    tokenizer_output = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in tokenizer_output.items()}

    # Get raw predictions
    with torch.no_grad():
        outputs = model(**inputs)
        raw_predictions = torch.argmax(outputs.logits, dim=2)

    # Align to original tokens
    word_ids = tokenizer_output.word_ids()
    aligned_preds = []
    previous_word_idx = None

    for j, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            aligned_preds.append(raw_predictions[0][j].item())
        previous_word_idx = word_idx

    # Apply post-processing
    corrected_preds = apply_post_processing_rules(
        aligned_preds,
        min_tokens_between_switches=min_tokens_between_switches
    )

    return corrected_preds


# ============================================================================
# FIXED POST-PROCESSING MODULE
# ============================================================================


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation"""
    test_file: str = 'dataset/preprocessed_augmented/test_segments_original.csv'
    # test_file: str = './test_segments.csv'
    tolerance: int = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model paths
    binary_model_path: str = './alloauto-presentation/web/model'
    mbert_model_path: str = './alloauto-segmentation-training/benchmark_models_standard/mbert_cased_baseline_standard_weighted_ner_24_10_params_alto/final_model'
    # mbert_model_path: str = './alloauto-segmentation-training/benchmark_models_standard/mbert_cased_baseline_standard_weighted_ner_23_10/final_model'
    xlmr_model_path: str = './alloauto-segmentation-training/benchmark_models_standard/xlm_roberta_baseline_standard_weighted_ner_24_10_params_alto/final_model'
    # xlmr_model_path: str = './alloauto-segmentation-training/benchmark_models_standard/xlm_roberta_baseline_standard_weighted_ner_23_10/final_model'
    # alto_model_path: str = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_allow_non_switch_test_train_and_fixed_loss_7_10_no_same_seqnence_simpler_loss/final_model'
    # alto_v2_model_path: str = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_allow_non_switch_test_train_and_fixed_loss_7_10_loss_segment_aware/final_model'
    # alto_3class_model_path: str = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_3class_NER_additive_loss_10_10/final_model'
    # alto_v2_additive_loss_model_path: str = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_additive_loss_10_10/final_model'
    # alto_simple_4_class_loss_model_path: str = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_simple_loss_4_class_NER_11_10/final_model'
    min_tokens_between_switches: int = 2

    model_names: List[str] = None

    cino_model_path: str = './alloauto-segmentation-training/benchmark_models_standard/cino_base_v2_baseline_standard_weighted_ner_24_10_params_alto/final_model'
    # cino_model_path: str = './alloauto-segmentation-training/benchmark_models_standard/cino_base_v2_baseline_standard_weighted_ner_23_10/final_model'
    # tibetan_roberta_model_path: str = './alloauto-segmentation-training/benchmark_models_standard/mbert_cased_baseline_standard_weighted_ner_24_10_params_alto/final_model'
    # tibetan_roberta_model_path: str = './alloauto-segmentation-training/benchmark_models_standard/tibetan_roberta_baseline_standard_weighted_ner_23_10/final_model'
    tibetan_roberta_model_path: str = './alloauto-segmentation-training/benchmark_models_standard/tibetan_roberta_baseline_standard_weighted_ner_24_10_params_alto/final_model'

    baseline_wylie_4class_path: str = './alloauto-segmentation-training/benchmark_models_standard/mbert_tibetan_wylie_baseline_standard_weighted_ner_24_10_params_alto/final_model'

    # Add these two lines after your existing model paths:
    simple_4class_model_path: str = './alloauto-segmentation-training/benchmark_models/simple_mBert_vanilla_benchmark_4_class_NER/final_model'
    simple_3class_model_path: str = './alloauto-segmentation-training/benchmark_models/simple_mBert_vanilla_benchmark_3_class_NER/final_model'

    mbert_3class_weighted_path: str = './alloauto-segmentation-training/benchmark_models/mBERT_3class_NER_additive_loss_10_10/final_model'

    alto_mul_with_seg_reward_path: str = './alloauto-segmentation-training/fine_tuned_ALTO_models/mbert_tibetan_wylie_ALTO_arch_23_10/final_model'
    alto_mul_no_seg_reward_path: str = './alloauto-segmentation-training/fine_tuned_ALTO_models/mbert_tibetan_wylie_no_seg_23_10/final_model'

    alto_additive_with_seg_reward_path: str = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_additive_loss_23_10/final_model'
    alto_additive_no_seg_reward_path: str = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_additive_loss_NO_SEG_REWARD_23_10/final_model'

    mBert_alto_loss_MUL_SEG_3_classes: str = './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mbert_cased_ALTO_MUL_SEG_arch_3class_24_10/final_model'
    mBert_alto_loss_MUL_SEG_4_classes: str = './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mbert_cased_ALTO_arch_23_10/final_model'

    mBert_alto_loss_ADDITIVE_SEG_3_classes: str = './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mBERT_additive_seg_loss_3class_23_10/final_model'
    mBert_alto_loss_ADDITIVE_SEG_4_classes: str = './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mBERT_vanilla_model_additive_loss_seg_23_10/final_model'

    mBert_simple_3_class: str = './alloauto-segmentation-training/benchmark_models_standard/simple_mBert_vanilla_benchmark_3_class_NER_23_10_no_early/final_model'
    mBert_simple_4_class: str = './alloauto-segmentation-training/benchmark_models_standard/simple_mBert_vanilla_benchmark_4_class_NER_23_10_no_early/final_model'

    def __post_init__(self):
        if self.model_names is None:
            self.model_names = [
                'Binary',
                'Random',

                'mBERT_WEIGHT_LOSS',
                'mBERT_WEIGHT_LOSS+PP',
                'XLM-R_WEIGHT_LOSS',
                'XLM-R_WEIGHT_LOSS+PP',
                'CINO_WEIGHT_LOSS',
                'CINO_WEIGHT_LOSS+PP',
                'Tibetan-RoBERTa_WEIGHT_LOSS',
                'Tibetan-RoBERTa_WEIGHT_LOSS+PP',
                'ALTO_MUL',
                'ALTO_MUL+PP',
                'ALTO_MUL_SEGMENT_REWARD',
                'ALTO_MUL_SEGMENT_REWARD+PP',
                'ALTO_ADDITIVE',
                'ALTO_ADDITIVE+PP',
                'ALTO_ADDITIVE_SEGMENT_REWARD',
                'ALTO_ADDITIVE_SEGMENT_REWARD+PP',
                'mBert_alto_MUL_SEG_3cls',
                'mBert_alto_MUL_SEG_3cls+PP',
                'mBert_alto_MUL_SEG_4cls',
                'mBert_alto_MUL_SEG_4cls+PP',
                'mBert_alto_ADD_SEG_3cls',
                'mBert_alto_ADD_SEG_3cls+PP',
                'mBert_alto_ADD_SEG_4cls',
                'mBert_alto_ADD_SEG_4cls+PP',
                'mBert_simple_3cls',
                'mBert_simple_3cls+PP',
                'mBert_simple_4cls',
                'mBert_simple_4cls+PP',
                'baseline_wylie_4cls',

            ]

# ============================================================================
# MODEL PROCESSORS
# ============================================================================
def remap_3class_to_4class(predictions_3class):
    """
    Remap 3-class predictions to 4-class format with MODE TRACKING.

    3-class: 0=NoSwitch, 1=Switch→Auto, 2=Switch→Allo
    4-class: 0=NonSwitch-Auto, 1=NonSwitch-Allo, 2=Switch→Auto, 3=Switch→Allo
    """
    remapped = []
    current_mode = 0  # Start in Auto mode

    for pred in predictions_3class:
        if pred == -100:
            remapped.append(-100)
        elif pred == 0:  # NoSwitch
            remapped.append(current_mode)  # Use current mode
        elif pred == 1:  # Switch→Auto
            remapped.append(2)
            current_mode = 0  # Now in Auto mode
        elif pred == 2:  # Switch→Allo
            remapped.append(3)
            current_mode = 1  # Now in Allo mode
        else:
            remapped.append(current_mode)

    return remapped
def process_random_model(tokens, avg_switches_per_segment=3.5, seed=42):
    """Random baseline"""
    np.random.seed(seed)
    n_tokens = len(tokens)
    n_switches = int(np.random.poisson(avg_switches_per_segment))

    if n_switches == 0:
        mode = np.random.choice([0, 1])
        return [mode] * n_tokens

    possible_positions = list(range(10, n_tokens - 10))
    if len(possible_positions) < n_switches:
        n_switches = len(possible_positions)

    if n_switches == 0 or len(possible_positions) == 0:
        mode = np.random.choice([0, 1])
        return [mode] * n_tokens

    switch_positions = sorted(np.random.choice(possible_positions, n_switches, replace=False))

    labels = []
    current_mode = np.random.choice([0, 1])
    switch_idx = 0

    for i in range(n_tokens):
        if switch_idx < len(switch_positions) and i == switch_positions[switch_idx]:
            if current_mode == 0:
                labels.append(3)
                current_mode = 1
            else:
                labels.append(2)
                current_mode = 0
            switch_idx += 1
        else:
            labels.append(current_mode)

    return labels


def process_binary_model_sentence_level(tokens, tokenizer, session):
    """Binary model at sentence level"""
    text = " ".join(tokens)
    chunks = re.split(r'\s*/+\s*', text)
    chunks = [c.strip() for c in chunks if c.strip()]

    if not chunks:
        chunks = [text]

    chunk_predictions = []

    for chunk in chunks:
        if not chunk:
            continue

        inputs = tokenizer(chunk, return_tensors="np", padding=False, truncation=True, max_length=512)
        onnx_inputs = {inp.name for inp in session.get_inputs()}
        filtered_inputs = {key: value for key, value in inputs.items() if key in onnx_inputs}

        outputs = session.run(None, filtered_inputs)
        logits = outputs[0]

        if len(logits.shape) == 2:
            logits = logits[0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        predicted_class = np.argmax(probs)
        chunk_predictions.append(predicted_class)

    token_labels = []
    token_idx = 0

    for chunk_idx, chunk in enumerate(chunks):
        chunk_tokens = chunk.split()
        chunk_class = chunk_predictions[chunk_idx] if chunk_idx < len(chunk_predictions) else 0

        is_switch = False
        if chunk_idx > 0 and chunk_idx < len(chunk_predictions):
            prev_class = chunk_predictions[chunk_idx - 1]
            curr_class = chunk_predictions[chunk_idx]
            if prev_class != curr_class:
                is_switch = True

        for i, token in enumerate(chunk_tokens):
            if token_idx < len(tokens):
                if i == 0 and is_switch:
                    label = 3 if chunk_class == 0 else 2
                else:
                    label = 1 if chunk_class == 0 else 0
                token_labels.append(label)
                token_idx += 1

    while len(token_labels) < len(tokens):
        token_labels.append(0)

    return token_labels[:len(tokens)]


def process_finetuned_model(tokens, tokenizer, model, device):
    """Fine-tuned model processing"""
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


def process_crf_model_fixed(tokens, tokenizer, model, device):
    """CRF model processing with proper alignment"""
    tokenizer_output = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    word_ids = tokenizer_output.word_ids()
    inputs = {k: v.to(device) for k, v in tokenizer_output.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        if 'predictions' in outputs:
            viterbi_predictions = outputs['predictions'][0]
        else:
            viterbi_predictions = torch.argmax(outputs['logits'], dim=2)[0].cpu().numpy()

    aligned_preds = []
    previous_word_idx = None
    viterbi_idx = 0

    if word_ids:
        for word_idx in word_ids:
            if word_idx is not None and word_idx != previous_word_idx:
                if viterbi_idx < len(viterbi_predictions):
                    aligned_preds.append(int(viterbi_predictions[viterbi_idx]))
                    viterbi_idx += 1
            previous_word_idx = word_idx

    while len(aligned_preds) < len(tokens):
        aligned_preds.append(0)

    return aligned_preds[:len(tokens)]


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_switch_detection_with_proximity(true_labels, pred_labels, tolerance=5):
    """Evaluate with proximity tolerance"""
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Find switch positions by type
    true_switches_to_auto = np.where(true_labels == 2)[0]
    true_switches_to_allo = np.where(true_labels == 3)[0]
    pred_switches_to_auto = np.where(pred_labels == 2)[0]
    pred_switches_to_allo = np.where(pred_labels == 3)[0]

    # Track matches
    matched_true_to_auto = set()
    matched_pred_to_auto = set()
    matched_true_to_allo = set()
    matched_pred_to_allo = set()

    exact_matches = 0
    proximity_matches = 0

    # Match predictions to true switches with tolerance
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

    # Calculate metrics
    total_true_switches = len(true_switches_to_auto) + len(true_switches_to_allo)
    total_pred_switches = len(pred_switches_to_auto) + len(pred_switches_to_allo)
    total_matches = exact_matches + proximity_matches

    proximity_precision = total_matches / total_pred_switches if total_pred_switches > 0 else 0
    proximity_recall = total_matches / total_true_switches if total_true_switches > 0 else 0
    proximity_f1 = 2 * proximity_precision * proximity_recall / (proximity_precision + proximity_recall) if (
                                                                                                                        proximity_precision + proximity_recall) > 0 else 0

    beta = 2
    proximity_fbeta2 = ((1 + beta ** 2) * proximity_precision * proximity_recall /
                        (beta ** 2 * proximity_precision + proximity_recall)) if (
                                                                                             proximity_precision + proximity_recall) > 0 else 0

    # Calculate mode accuracy for non-switch tokens
    non_switch_true = (true_labels <= 1)
    non_switch_pred = (pred_labels <= 1)

    if non_switch_true.sum() > 0:
        # Mode accuracy: correct mode prediction for non-switch tokens
        mode_true = true_labels[non_switch_true]
        mode_pred = pred_labels[non_switch_true & non_switch_pred]

        if len(mode_pred) > 0 and len(mode_true) > 0:
            # Align the arrays properly
            min_len = min(len(mode_true), len(np.where(non_switch_true & non_switch_pred)[0]))
            if min_len > 0:
                mode_accuracy = np.mean(true_labels[non_switch_true][:min_len] ==
                                        pred_labels[non_switch_true][:min_len])
            else:
                mode_accuracy = 0
        else:
            mode_accuracy = 0
    else:
        mode_accuracy = 0

    return {
        'proximity_precision': proximity_precision,
        'proximity_recall': proximity_recall,
        'proximity_f1': proximity_f1,
        'proximity_fbeta2': proximity_fbeta2,
        'mode_accuracy': mode_accuracy,
        'true_switches': total_true_switches,
        'pred_switches': total_pred_switches,
        'total_matches': total_matches,
        'exact_matches': exact_matches,
        'proximity_matches': proximity_matches,
    }


def process_simple_3class_model(tokens, tokenizer, model, device):
    """
    Process 3-class model and remap to 4-class format
    """
    # Get raw 3-class predictions
    raw_preds = process_finetuned_model(tokens, tokenizer, model, device)

    # Remap to 4-class format
    remapped_preds = remap_3class_to_4class(raw_preds)

    return remapped_preds
# ============================================================================
# MODEL MANAGER
# ============================================================================
class ModelManager:
    """Manages all models including simple 4-class and 3-class"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.models = {}
        self.tokenizers = {}

    def load_all_models(self):
        print("Loading all models...")

        # Binary
        self.tokenizers['Binary'] = AutoTokenizer.from_pretrained(self.config.binary_model_path)
        self.models['Binary'] = ort.InferenceSession(f'{self.config.binary_model_path}/onnx/model.onnx')

        # mBERT
        self.tokenizers['mBERT_WEIGHT_LOSS'] = AutoTokenizer.from_pretrained(self.config.mbert_model_path)
        self.models['mBERT_WEIGHT_LOSS'] = AutoModelForTokenClassification.from_pretrained(self.config.mbert_model_path).eval().to(
            self.config.device)
        self.tokenizers['mBERT_WEIGHT_LOSS+PP'] = self.tokenizers['mBERT_WEIGHT_LOSS'];
        # self.models['mBERT_ADDITIVE_LOSS+PP'] = self.models['mBERT_ADDITIVE_LOSS']self.tokenizers['mBERT_ADDITIVE_LOSS'] = AutoTokenizer.from_pretrained(self.config.mbert_model_path)
        # self.models['mBERT_ADDITIVE_LOSS'] = AutoModelForTokenClassification.from_pretrained(self.config.mbert_model_path).eval().to(
        #     self.config.device)
        # self.tokenizers['mBERT_ADDITIVE_LOSS+PP'] = self.tokenizers['mBERT_ADDITIVE_LOSS'];
        # self.models['mBERT_ADDITIVE_LOSS+PP'] = self.models['mBERT_ADDITIVE_LOSS']

        # XLM-R
        self.tokenizers['XLM-R_WEIGHT_LOSS'] = AutoTokenizer.from_pretrained(self.config.xlmr_model_path)
        self.models['XLM-R_WEIGHT_LOSS'] = AutoModelForTokenClassification.from_pretrained(self.config.xlmr_model_path).eval().to(
            self.config.device)
        self.tokenizers['XLM-R_WEIGHT_LOSS+PP'] = self.tokenizers['XLM-R_WEIGHT_LOSS'];
        self.models['XLM-R_WEIGHT_LOSS+PP'] = self.models['XLM-R_WEIGHT_LOSS']

        # NEW: CINO
        self.tokenizers['CINO_WEIGHT_LOSS'] = AutoTokenizer.from_pretrained(self.config.cino_model_path)

        # self.tokenizers['CINO'] = AutoTokenizer.from_pretrained(self.config.cino_model_path, use_fast=False,
        #                                                         add_prefix_space=True)
        self.models['CINO_WEIGHT_LOSS'] = AutoModelForTokenClassification.from_pretrained(self.config.cino_model_path).eval().to(
            self.config.device)
        self.tokenizers['CINO_WEIGHT_LOSS+PP'] = self.tokenizers['CINO_WEIGHT_LOSS'];
        self.models['CINO_WEIGHT_LOSS+PP'] = self.models['CINO_WEIGHT_LOSS']

        # NEW: Tibetan-RoBERTa
        self.tokenizers['Tibetan-RoBERTa_WEIGHT_LOSS'] = AutoTokenizer.from_pretrained(self.config.tibetan_roberta_model_path,
                                                                           use_fast=True, add_prefix_space=True)
        self.models['Tibetan-RoBERTa_WEIGHT_LOSS'] = AutoModelForTokenClassification.from_pretrained(
            self.config.tibetan_roberta_model_path).eval().to(self.config.device)
        self.tokenizers['Tibetan-RoBERTa_WEIGHT_LOSS+PP'] = self.tokenizers['Tibetan-RoBERTa_WEIGHT_LOSS']
        self.models['Tibetan-RoBERTa_WEIGHT_LOSS+PP'] = self.models['Tibetan-RoBERTa_WEIGHT_LOSS']

        # ALTO MUL WITHOUT SEGMENTS REWARD
        self.tokenizers['ALTO_MUL'] = AutoTokenizer.from_pretrained(self.config.alto_mul_no_seg_reward_path)
        self.models['ALTO_MUL'] = AutoModelForTokenClassification.from_pretrained(self.config.alto_mul_no_seg_reward_path).eval().to(
            self.config.device)
        self.tokenizers['ALTO_MUL+PP'] = self.tokenizers['ALTO_MUL']
        self.models['ALTO_MUL+PP'] = self.models['ALTO_MUL']

        #ALTO MUL WITH SEGMENTS
        self.tokenizers['ALTO_MUL_SEGMENT_REWARD'] = AutoTokenizer.from_pretrained(
            self.config.alto_mul_with_seg_reward_path)
        self.models['ALTO_MUL_SEGMENT_REWARD'] = AutoModelForTokenClassification.from_pretrained(
            self.config.alto_mul_with_seg_reward_path).eval().to(
            self.config.device)
        self.tokenizers['ALTO_MUL_SEGMENT_REWARD+PP'] = self.tokenizers['ALTO_MUL_SEGMENT_REWARD']
        self.models['ALTO_MUL_SEGMENT_REWARD+PP'] = self.models['ALTO_MUL_SEGMENT_REWARD']

        # ALTO ADDITIVE_LOSS WITHOUT SEGMENTS REWARD
        self.tokenizers['ALTO_ADDITIVE'] = AutoTokenizer.from_pretrained(
            self.config.alto_additive_no_seg_reward_path)
        self.models['ALTO_ADDITIVE'] = AutoModelForTokenClassification.from_pretrained(
            self.config.alto_additive_no_seg_reward_path).eval().to(self.config.device)
        self.tokenizers['ALTO_ADDITIVE+PP'] = self.tokenizers['ALTO_ADDITIVE']
        self.models['ALTO_ADDITIVE+PP'] = self.models['ALTO_ADDITIVE']

        #ALTO ADDITIVE_LOSS WITH SEGMENTS REWARD
        self.tokenizers['ALTO_ADDITIVE_SEGMENT_REWARD'] = AutoTokenizer.from_pretrained(self.config.alto_additive_with_seg_reward_path)
        self.models['ALTO_ADDITIVE_SEGMENT_REWARD'] = AutoModelForTokenClassification.from_pretrained(
            self.config.alto_additive_with_seg_reward_path).eval().to(self.config.device)
        self.tokenizers['ALTO_ADDITIVE_SEGMENT_REWARD+PP'] = self.tokenizers['ALTO_ADDITIVE_SEGMENT_REWARD']
        self.models['ALTO_ADDITIVE_SEGMENT_REWARD+PP'] = self.models['ALTO_ADDITIVE_SEGMENT_REWARD']

        if 'mBert_alto_MUL_SEG_3cls' in self.config.model_names or 'mBert_alto_MUL_SEG_3cls+PP' in self.config.model_names:
            print("  Loading mBert_alto_MUL_SEG_3cls...")
            self.tokenizers['mBert_alto_MUL_SEG_3cls'] = AutoTokenizer.from_pretrained(
                self.config.mBert_alto_loss_MUL_SEG_3_classes
            )
            self.models['mBert_alto_MUL_SEG_3cls'] = AutoModelForTokenClassification.from_pretrained(
                self.config.mBert_alto_loss_MUL_SEG_3_classes
            ).to(self.config.device)
            self.models['mBert_alto_MUL_SEG_3cls'].eval()

            # ADD THIS: Load mBert_alto_MUL_SEG_4cls
        if 'mBert_alto_MUL_SEG_4cls' in self.config.model_names or 'mBert_alto_MUL_SEG_4cls+PP' in self.config.model_names:
            print("  Loading mBert_alto_MUL_SEG_4cls...")
            self.tokenizers['mBert_alto_MUL_SEG_4cls'] = AutoTokenizer.from_pretrained(
                self.config.mBert_alto_loss_MUL_SEG_4_classes
            )
            self.models['mBert_alto_MUL_SEG_4cls'] = AutoModelForTokenClassification.from_pretrained(
                self.config.mBert_alto_loss_MUL_SEG_4_classes
            ).to(self.config.device)
            self.models['mBert_alto_MUL_SEG_4cls'].eval()

            # ADD THIS: Load mBert_alto_ADD_SEG_3cls
        if 'mBert_alto_ADD_SEG_3cls' in self.config.model_names or 'mBert_alto_ADD_SEG_3cls+PP' in self.config.model_names:
            print("  Loading mBert_alto_ADD_SEG_3cls...")
            self.tokenizers['mBert_alto_ADD_SEG_3cls'] = AutoTokenizer.from_pretrained(
                self.config.mBert_alto_loss_ADDITIVE_SEG_3_classes
            )
            self.models['mBert_alto_ADD_SEG_3cls'] = AutoModelForTokenClassification.from_pretrained(
                self.config.mBert_alto_loss_ADDITIVE_SEG_3_classes
            ).to(self.config.device)
            self.models['mBert_alto_ADD_SEG_3cls'].eval()

            # ADD THIS: Load mBert_alto_ADD_SEG_4cls
        if 'mBert_alto_ADD_SEG_4cls' in self.config.model_names or 'mBert_alto_ADD_SEG_4cls+PP' in self.config.model_names:
            print("  Loading mBert_alto_ADD_SEG_4cls...")
            self.tokenizers['mBert_alto_ADD_SEG_4cls'] = AutoTokenizer.from_pretrained(
                self.config.mBert_alto_loss_ADDITIVE_SEG_4_classes
            )
            self.models['mBert_alto_ADD_SEG_4cls'] = AutoModelForTokenClassification.from_pretrained(
                self.config.mBert_alto_loss_ADDITIVE_SEG_4_classes
            ).to(self.config.device)
            self.models['mBert_alto_ADD_SEG_4cls'].eval()

            # ADD THIS: Load mBert_simple_3cls
        if 'mBert_simple_3cls' in self.config.model_names or 'mBert_simple_3cls+PP' in self.config.model_names:
            print("  Loading mBert_simple_3cls...")
            self.tokenizers['mBert_simple_3cls'] = AutoTokenizer.from_pretrained(
                self.config.mBert_simple_3_class
            )
            self.models['mBert_simple_3cls'] = AutoModelForTokenClassification.from_pretrained(
                self.config.mBert_simple_3_class
            ).to(self.config.device)
            self.models['mBert_simple_3cls'].eval()

            # ADD THIS: Load mBert_simple_4cls
        if 'mBert_simple_4cls' in self.config.model_names or 'mBert_simple_4cls+PP' in self.config.model_names:
            print("  Loading mBert_simple_4cls...")
            self.tokenizers['mBert_simple_4cls'] = AutoTokenizer.from_pretrained(
                self.config.mBert_simple_4_class
            )
            self.models['mBert_simple_4cls'] = AutoModelForTokenClassification.from_pretrained(
                self.config.mBert_simple_4_class
            ).to(self.config.device)
            self.models['mBert_simple_4cls'].eval()

        if 'baseline_wylie_4cls' in self.config.model_names or 'baseline_wylie_4cls+PP' in self.config.model_names:
            print("  Loading baseline_wylie_4cls...")
            self.tokenizers['baseline_wylie_4cls'] = AutoTokenizer.from_pretrained(
                self.config.baseline_wylie_4class_path
            )
            self.models['baseline_wylie_4cls'] = AutoModelForTokenClassification.from_pretrained(
                self.config.baseline_wylie_4class_path
            ).to(self.config.device)
            self.models['baseline_wylie_4cls'].eval()

        print(f"✓ Loaded {len(self.models)} unique models")
        return self

        ######### untill now!!!
        # ALTO-4Class
        # self.tokenizers['ALTO-4Class'] = AutoTokenizer.from_pretrained(self.config.alto_simple_4_class_loss_model_path)
        # self.models['ALTO-4Class'] = AutoModelForTokenClassification.from_pretrained(
        #     self.config.alto_focal_model_path).eval().to(self.config.device)
        # self.tokenizers['ALTO-4Class+PP'] = self.tokenizers['ALTO-4Class']
        # self.models['ALTO-4Class+PP'] = self.models['ALTO-4Class']
        #
        # # ADD THIS SECTION - ALTO 3-Class
        # print("Loading ALTO 3-Class model...")
        # self.tokenizers['ALTO-3Class_ADDITIVE_LOSS'] = AutoTokenizer.from_pretrained(self.config.alto_3class_model_path)
        # self.models['ALTO-3Class_ADDITIVE_LOSS'] = AutoModelForTokenClassification.from_pretrained(
        #     self.config.alto_3class_model_path).eval().to(self.config.device)
        # self.tokenizers['ALTO-3Class_ADDITIVE_LOSS+PP'] = self.tokenizers['ALTO-3Class_ADDITIVE_LOSS']
        # self.models['ALTO-3Class_ADDITIVE_LOSS+PP'] = self.models['ALTO-3Class_ADDITIVE_LOSS']
        # print("✓ Loaded ALTO 3-Class model")
        #
        # self.tokenizers['Simple-4Class'] = AutoTokenizer.from_pretrained(self.config.simple_4class_model_path)
        # self.models['Simple-4Class'] = AutoModelForTokenClassification.from_pretrained(
        #     self.config.simple_4class_model_path).eval().to(self.config.device)
        # self.tokenizers['Simple-4Class+PP'] = self.tokenizers['Simple-4Class']
        # self.models['Simple-4Class+PP'] = self.models['Simple-4Class']
        #
        # # Simple 3-class model
        # self.tokenizers['Simple-3Class'] = AutoTokenizer.from_pretrained(self.config.simple_3class_model_path)
        # self.models['Simple-3Class'] = AutoModelForTokenClassification.from_pretrained(
        #     self.config.simple_3class_model_path).eval().to(self.config.device)
        # self.tokenizers['Simple-3Class+PP'] = self.tokenizers['Simple-3Class']
        # self.models['Simple-3Class+PP'] = self.models['Simple-3Class']
        #
        # # Simple 4-class model
        # print("Loading Simple 4-class model...")
        # self.tokenizers['Simple-4Class'] = AutoTokenizer.from_pretrained(self.config.simple_4class_model_path)
        # self.models['Simple-4Class'] = AutoModelForTokenClassification.from_pretrained(
        #     self.config.simple_4class_model_path).eval().to(self.config.device)
        # self.tokenizers['Simple-4Class+PP'] = self.tokenizers['Simple-4Class']
        # self.models['Simple-4Class+PP'] = self.models['Simple-4Class']
        #
        # print("Loading mBERT 3-Class Weighted model...")
        # self.tokenizers['mBERT-3Class_ADDITIVE_LOSS'] = AutoTokenizer.from_pretrained(self.config.mbert_3class_weighted_path)
        # self.models['mBERT-3Class_ADDITIVE_LOSS'] = AutoModelForTokenClassification.from_pretrained(
        #     self.config.mbert_3class_weighted_path).eval().to(self.config.device)
        # self.tokenizers['mBERT-3Class_ADDITIVE_LOSS+PP'] = self.tokenizers['mBERT-3Class_ADDITIVE_LOSS']
        # self.models['mBERT-3Class_ADDITIVE_LOSS+PP'] = self.models['mBERT-3Class_ADDITIVE_LOSS']

        print("✓ Loaded mBERT 3-Class Weighted model")

        print("✓ Loaded Simple 4-class and 3-class models")
        return self


    def predict_single_segment(self, model_name: str, tokens: List[str], seed: int = 42) -> List[int]:
        """Predict for a single segment with PP variants"""
        if model_name == 'Random':
            return process_random_model(tokens, seed=seed)

        elif model_name == 'Binary':
            preds = process_binary_model_sentence_level(
                tokens, self.tokenizers['Binary'], self.models['Binary']
            )

        # elif model_name == 'mBERT_ADDITIVE_LOSS':
        #     preds = process_finetuned_model(
        #         tokens, self.tokenizers['mBERT_ADDITIVE_LOSS'], self.models['mBERT_ADDITIVE_LOSS'], self.config.device
        #     )
        #
        # elif model_name == 'mBERT_ADDITIVE_LOSS+PP':
        #     raw_preds = process_finetuned_model(
        #         tokens, self.tokenizers['mBERT_ADDITIVE_LOSS'], self.models['mBERT_ADDITIVE_LOSS'], self.config.device
        #     )
        #     preds = apply_post_processing_rules(
        #         raw_preds[:len(tokens)],
        #         min_tokens_between_switches=self.config.min_tokens_between_switches
        #     )


        ###
        elif model_name == 'mBERT_WEIGHT_LOSS':
            preds = process_finetuned_model(
                tokens, self.tokenizers['mBERT_WEIGHT_LOSS'], self.models['mBERT_WEIGHT_LOSS'], self.config.device
            )

        elif model_name == 'mBERT_WEIGHT_LOSS+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['mBERT_WEIGHT_LOSS'], self.models['mBERT_WEIGHT_LOSS'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'XLM-R_WEIGHT_LOSS':
            preds = process_finetuned_model(
                tokens, self.tokenizers['XLM-R_WEIGHT_LOSS'], self.models['XLM-R_WEIGHT_LOSS'], self.config.device
            )

        elif model_name == 'XLM-R_WEIGHT_LOSS+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['XLM-R_WEIGHT_LOSS'], self.models['XLM-R_WEIGHT_LOSS'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )
            print(verify_post_processing_rules(preds))  # should show empty lists for adjacent/insufficient/invalid

        elif model_name == 'CINO_WEIGHT_LOSS':
            preds = process_finetuned_model(tokens, self.tokenizers['CINO_WEIGHT_LOSS'], self.models['CINO_WEIGHT_LOSS'], self.config.device)

        elif model_name == 'CINO_WEIGHT_LOSS+PP':
            raw = process_finetuned_model(tokens, self.tokenizers['CINO_WEIGHT_LOSS'], self.models['CINO_WEIGHT_LOSS'], self.config.device)
            preds = apply_post_processing_rules(raw[:len(tokens)],
                                                min_tokens_between_switches=self.config.min_tokens_between_switches)

        elif model_name == 'Tibetan-RoBERTa_WEIGHT_LOSS':
            preds = process_finetuned_model(tokens, self.tokenizers['Tibetan-RoBERTa_WEIGHT_LOSS'], self.models['Tibetan-RoBERTa_WEIGHT_LOSS'],
                                            self.config.device)

        elif model_name == 'Tibetan-RoBERTa_WEIGHT_LOSS+PP':
            raw = process_finetuned_model(tokens, self.tokenizers['Tibetan-RoBERTa_WEIGHT_LOSS'], self.models['Tibetan-RoBERTa_WEIGHT_LOSS'],
                                          self.config.device)
            preds = apply_post_processing_rules(raw[:len(tokens)],
                                                min_tokens_between_switches=self.config.min_tokens_between_switches)

        elif model_name == 'ALTO_MUL':
            preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO_MUL'], self.models['ALTO_MUL'], self.config.device
            )


        elif model_name == 'ALTO_MUL+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO_MUL'], self.models['ALTO_MUL'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )


        elif model_name == 'ALTO_MUL_SEGMENT_REWARD':
            preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO_MUL_SEGMENT_REWARD'], self.models['ALTO_MUL_SEGMENT_REWARD'], self.config.device
            )

        elif model_name == 'ALTO_MUL_SEGMENT_REWARD+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO_MUL_SEGMENT_REWARD'], self.models['ALTO_MUL_SEGMENT_REWARD'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'ALTO_ADDITIVE':
            preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO_ADDITIVE'], self.models['ALTO_ADDITIVE'], self.config.device
            )

        elif model_name == 'ALTO_ADDITIVE+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO_ADDITIVE'], self.models['ALTO_ADDITIVE'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'ALTO_ADDITIVE_SEGMENT_REWARD':
            preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO_ADDITIVE_SEGMENT_REWARD'], self.models['ALTO_ADDITIVE_SEGMENT_REWARD'], self.config.device
            )

        elif model_name == 'ALTO_ADDITIVE_SEGMENT_REWARD+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO_ADDITIVE_SEGMENT_REWARD'], self.models['ALTO_ADDITIVE_SEGMENT_REWARD'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'ALTO-4Class':
            preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO-4Class'], self.models['ALTO-4Class'], self.config.device
            )

        elif model_name == 'ALTO-4Class+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['ALTO-4Class'], self.models['ALTO-4Class'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        # ADD THIS SECTION - ALTO 3-Class predictions
        elif model_name == 'ALTO-3Class_ADDITIVE_LOSS':
            # Get 3-class predictions and remap to 4-class
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['ALTO-3Class_ADDITIVE_LOSS'],
                self.models['ALTO-3Class_ADDITIVE_LOSS'], self.config.device
            )
            preds = remap_3class_to_4class(raw_preds_3class)

        elif model_name == 'ALTO-3Class_ADDITIVE_LOSS+PP':
            # Get 3-class predictions, remap, then apply PP
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['ALTO-3Class_ADDITIVE_LOSS'],
                self.models['ALTO-3Class_ADDITIVE_LOSS'], self.config.device
            )
            remapped = remap_3class_to_4class(raw_preds_3class)
            preds = apply_post_processing_rules(
                remapped[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'Simple-4Class':
            preds = process_finetuned_model(
                tokens, self.tokenizers['Simple-4Class'],
                self.models['Simple-4Class'], self.config.device
            )

        elif model_name == 'Simple-4Class+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['Simple-4Class'],
                self.models['Simple-4Class'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'Simple-3Class':
            # Get 3-class predictions and remap to 4-class
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['Simple-3Class'],
                self.models['Simple-3Class'], self.config.device
            )
            preds = remap_3class_to_4class(raw_preds_3class)

        elif model_name == 'Simple-3Class+PP':
            # Get 3-class predictions, remap, then apply PP
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['Simple-3Class'],
                self.models['Simple-3Class'], self.config.device
            )
            remapped = remap_3class_to_4class(raw_preds_3class)
            preds = apply_post_processing_rules(
                remapped[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )
        elif model_name == 'mBERT-3Class_ADDITIVE_LOSS':
            # Get 3-class predictions and remap to 4-class
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['mBERT-3Class_ADDITIVE_LOSS'],
                self.models['mBERT-3Class_ADDITIVE_LOSS'], self.config.device
            )
            preds = remap_3class_to_4class(raw_preds_3class)

        elif model_name == 'mBERT-3Class_ADDITIVE_LOSS+PP':
            # Get 3-class predictions, remap, then apply PP
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['mBERT-3Class_ADDITIVE_LOSS'],
                self.models['mBERT-3Class_ADDITIVE_LOSS'], self.config.device
            )
            remapped = remap_3class_to_4class(raw_preds_3class)
            preds = apply_post_processing_rules(
                remapped[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        #########
        elif model_name == 'mBert_alto_MUL_SEG_3cls':
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['mBert_alto_MUL_SEG_3cls'],
                self.models['mBert_alto_MUL_SEG_3cls'], self.config.device
            )
            preds = remap_3class_to_4class(raw_preds_3class)

        elif model_name == 'mBert_alto_MUL_SEG_3cls+PP':
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['mBert_alto_MUL_SEG_3cls'],
                self.models['mBert_alto_MUL_SEG_3cls'], self.config.device
            )
            remapped = remap_3class_to_4class(raw_preds_3class)
            preds = apply_post_processing_rules(
                remapped[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'mBert_alto_MUL_SEG_4cls':
            preds = process_finetuned_model(
                tokens, self.tokenizers['mBert_alto_MUL_SEG_4cls'],
                self.models['mBert_alto_MUL_SEG_4cls'], self.config.device
            )

        elif model_name == 'mBert_alto_MUL_SEG_4cls+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['mBert_alto_MUL_SEG_4cls'],
                self.models['mBert_alto_MUL_SEG_4cls'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'mBert_alto_ADD_SEG_3cls':
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['mBert_alto_ADD_SEG_3cls'],
                self.models['mBert_alto_ADD_SEG_3cls'], self.config.device
            )
            preds = remap_3class_to_4class(raw_preds_3class)

        elif model_name == 'mBert_alto_ADD_SEG_3cls+PP':
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['mBert_alto_ADD_SEG_3cls'],
                self.models['mBert_alto_ADD_SEG_3cls'], self.config.device
            )
            remapped = remap_3class_to_4class(raw_preds_3class)
            preds = apply_post_processing_rules(
                remapped[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'mBert_alto_ADD_SEG_4cls':
            preds = process_finetuned_model(
                tokens, self.tokenizers['mBert_alto_ADD_SEG_4cls'],
                self.models['mBert_alto_ADD_SEG_4cls'], self.config.device
            )

        elif model_name == 'mBert_alto_ADD_SEG_4cls+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['mBert_alto_ADD_SEG_4cls'],
                self.models['mBert_alto_ADD_SEG_4cls'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'mBert_simple_3cls':
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['mBert_simple_3cls'],
                self.models['mBert_simple_3cls'], self.config.device
            )
            preds = remap_3class_to_4class(raw_preds_3class)

        elif model_name == 'mBert_simple_3cls+PP':
            raw_preds_3class = process_finetuned_model(
                tokens, self.tokenizers['mBert_simple_3cls'],
                self.models['mBert_simple_3cls'], self.config.device
            )
            remapped = remap_3class_to_4class(raw_preds_3class)
            preds = apply_post_processing_rules(
                remapped[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )

        elif model_name == 'mBert_simple_4cls':
            preds = process_finetuned_model(
                tokens, self.tokenizers['mBert_simple_4cls'],
                self.models['mBert_simple_4cls'], self.config.device
            )

        elif model_name == 'mBert_simple_4cls+PP':
            raw_preds = process_finetuned_model(
                tokens, self.tokenizers['mBert_simple_4cls'],
                self.models['mBert_simple_4cls'], self.config.device
            )
            preds = apply_post_processing_rules(
                raw_preds[:len(tokens)],
                min_tokens_between_switches=self.config.min_tokens_between_switches
            )
        elif model_name == 'baseline_wylie_4cls':
            preds = process_finetuned_model(
                tokens, self.tokenizers['baseline_wylie_4cls'],
                self.models['baseline_wylie_4cls'], self.config.device
            )



        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Ensure length match
        if len(preds) > len(tokens):
            preds = preds[:len(tokens)]
        elif len(preds) < len(tokens):
            last_pred = preds[-1] if preds else 0
            preds.extend([last_pred] * (len(tokens) - len(preds)))

        return preds


# ============================================================================
# COMPREHENSIVE EVALUATOR
# ============================================================================

class ComprehensiveEvaluator:
    """Evaluates on all segments, switch segments, and non-switch segments"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {
            'all': {},
            'switch': {},
            'no_switch': {}
        }

    def evaluate_all_models(self, test_df: pd.DataFrame, model_manager: ModelManager):
        """Evaluate all models on different segment types"""
        print(f"\nEvaluating on {len(test_df)} segments...")

        # Split segments into categories
        switch_segments = []
        no_switch_segments = []

        for idx, row in test_df.iterrows():
            tokens = row['tokens'].split()
            true_labels = [int(l) for l in row['labels'].split(',')]

            has_switch = any(l in [2, 3] for l in true_labels)

            if has_switch:
                switch_segments.append(idx)
            else:
                no_switch_segments.append(idx)

        print(f"  Segments with switches: {len(switch_segments)}")
        print(f"  Segments without switches: {len(no_switch_segments)}")
        print(f"  Total segments: {len(test_df)}")

        # Initialize accumulators for each category
        segment_metrics = {
            'all': {name: [] for name in self.config.model_names},
            'switch': {name: [] for name in self.config.model_names},
            'no_switch': {name: [] for name in self.config.model_names}
        }
        diag_accum = {
            name: {
                "pair_count": 0,
                "gaps": [],
                "adjacent_pairs": 0,
                "single_gap_pairs": 0,
                "same_type_pairs": 0,
                "max_same_type_run": 0,
            } for name in self.config.model_names
        }

        # Process each segment
        for idx, row in test_df.iterrows():
            if idx % 50 == 0:
                print(f"  Processing segment {idx}/{len(test_df)}...")

            tokens = row['tokens'].split()
            true_labels = [int(l) for l in row['labels'].split(',')]

            # Ensure alignment
            min_len = min(len(tokens), len(true_labels))
            tokens = tokens[:min_len]
            true_labels = true_labels[:min_len]

            # Determine segment type
            has_switch = any(l in [2, 3] for l in true_labels)

            # Get predictions from each model
            for model_name in self.config.model_names:
                preds = model_manager.predict_single_segment(model_name, tokens, seed=42 + idx)

                # Ensure alignment
                if len(preds) != len(true_labels):
                    min_len = min(len(preds), len(true_labels))
                    preds = preds[:min_len]
                    segment_true = true_labels[:min_len]
                else:
                    segment_true = true_labels

                # Evaluate this segment
                segment_result = evaluate_switch_detection_with_proximity(
                    segment_true, preds, self.config.tolerance
                )
                diag = switch_sequence_diagnostics(preds)
                acc = diag_accum[model_name]
                acc["pair_count"] += diag["pair_count"]
                acc["adjacent_pairs"] += diag["adjacent_pairs"]
                acc["single_gap_pairs"] += diag["single_gap_pairs"]
                acc["same_type_pairs"] += diag["same_type_pairs"]
                # keep max run across segments
                acc["max_same_type_run"] = max(acc["max_same_type_run"], diag["max_same_type_run"])
                # collect gaps for mean/median/min/max
                acc["gaps"].extend(diag["gaps"])

                # Add to appropriate categories
                segment_metrics['all'][model_name].append(segment_result)

                if has_switch:
                    segment_metrics['switch'][model_name].append(segment_result)
                else:
                    segment_metrics['no_switch'][model_name].append(segment_result)

        # Aggregate results for each category
        print("\nAggregating results...")

        for category in ['all', 'switch', 'no_switch']:
            print(f"  Aggregating {category} segments...")

            for model_name in self.config.model_names:
                metrics = segment_metrics[category][model_name]

                if not metrics:
                    # No segments in this category
                    self.results[category][model_name] = {
                        'proximity_precision': 0,
                        'proximity_recall': 0,
                        'proximity_f1': 0,
                        'proximity_fbeta2': 0,
                        'mode_accuracy': 0,
                        'true_switches': 0,
                        'pred_switches': 0,
                        'total_matches': 0,
                        'segment_count': 0
                    }
                    continue

                # Aggregate metrics
                total_true_switches = sum(m['true_switches'] for m in metrics)
                total_pred_switches = sum(m['pred_switches'] for m in metrics)
                total_matches = sum(m['total_matches'] for m in metrics)
                exact_matches = sum(m['exact_matches'] for m in metrics)
                proximity_matches = sum(m['proximity_matches'] for m in metrics)

                # Calculate overall metrics
                precision = total_matches / total_pred_switches if total_pred_switches > 0 else 0
                recall = total_matches / total_true_switches if total_true_switches > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                beta = 2
                fbeta2 = ((1 + beta ** 2) * precision * recall /
                          (beta ** 2 * precision + recall)) if (precision + recall) > 0 else 0

                # Average mode accuracy
                avg_mode_accuracy = np.mean([m['mode_accuracy'] for m in metrics])

                self.results[category][model_name] = {
                    'proximity_precision': precision,
                    'proximity_recall': recall,
                    'proximity_f1': f1,
                    'proximity_fbeta2': fbeta2,
                    'mode_accuracy': avg_mode_accuracy,
                    'true_switches': total_true_switches,
                    'pred_switches': total_pred_switches,
                    'total_matches': total_matches,
                    'exact_matches': exact_matches,
                    'proximity_matches': proximity_matches,
                    'segment_count': len(metrics)
                }
        self.switch_diagnostics = {}
        for name, acc in diag_accum.items():
            gaps = acc["gaps"]
            pair_count = acc["pair_count"]
            self.switch_diagnostics[name] = {
                "pairs": pair_count,
                "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
                "median_gap": float(np.median(gaps)) if gaps else 0.0,
                "min_gap": int(np.min(gaps)) if gaps else 0,
                "max_gap": int(np.max(gaps)) if gaps else 0,
                "adjacent_pairs": acc["adjacent_pairs"],  # gaps == 0
                "single_gap_pairs": acc["single_gap_pairs"],  # gaps == 1
                "same_type_pairs": acc["same_type_pairs"],  # e.g., 3,3 or 2,2
                "max_same_type_run": acc["max_same_type_run"],  # longest run length
                "pct_adjacent": (acc["adjacent_pairs"] / pair_count) if pair_count else 0.0,
                "pct_single_gap": (acc["single_gap_pairs"] / pair_count) if pair_count else 0.0,
                "pct_same_type_consec": (acc["same_type_pairs"] / pair_count) if pair_count else 0.0,
            }
        return self.results, segment_metrics, self.switch_diagnostics


# ============================================================================
# COMPREHENSIVE REPORTER
# ============================================================================

class ComprehensiveReporter:
    """Reports results for all segment categories"""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _print_winner_summary(self, results: Dict):
        """Print summary of which models win the most metrics across categories"""
        print("\n" + "=" * 200)
        print("BEST PERFORMER SUMMARY ACROSS ALL CATEGORIES")
        print("=" * 200)

        # Count wins per model
        win_counts = {model: 0 for model in self.config.model_names}
        win_details = {model: [] for model in self.config.model_names}

        categories = [
            ('ALL', results['all']),
            ('SWITCH', results['switch']),
            ('NO_SWITCH', results['no_switch'])
        ]

        metrics = [
            ('F-β(2)', 'proximity_fbeta2'),
            ('Precision', 'proximity_precision'),
            ('Recall', 'proximity_recall'),
            ('F1', 'proximity_f1'),
            ('Mode Accuracy', 'mode_accuracy')
        ]

        for cat_name, cat_results in categories:
            for metric_name, metric_key in metrics:
                # Skip mode accuracy for switch segments
                if metric_key == 'mode_accuracy' and cat_name == 'SWITCH':
                    continue

                # Find best value and models
                best_val = 0
                best_models = []

                for model in self.config.model_names:
                    if model in cat_results:
                        val = cat_results[model].get(metric_key, 0)
                        if val > best_val:
                            best_val = val
                            best_models = [model]
                        elif val == best_val and val > 0:
                            best_models.append(model)

                # Award wins
                for model in best_models:
                    win_counts[model] += 1
                    win_details[model].append(f"{cat_name}-{metric_name}")

        # Sort by win count
        sorted_models = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'Model':<22} {'Wins':<8} {'Winning Metrics'}")
        print("-" * 80)

        for model, wins in sorted_models:
            if wins > 0:
                details = ", ".join(win_details[model][:3])  # Show first 3
                if len(win_details[model]) > 3:
                    details += f", ... (+{len(win_details[model]) - 3} more)"
                print(f"{model:<22} {wins:<8} {details}")
    def print_all_results(self, results: Dict, switch_diagnostics):
        """Print comprehensive results for all categories"""

        # 1. Overall results (all segments)
        self._print_category_table(results['all'], "ALL SEGMENTS", include_mode_accuracy=True)

        # 2. Switch segments only
        self._print_category_table(results['switch'], "SEGMENTS WITH SWITCHES ONLY", include_mode_accuracy=False)

        # 3. Non-switch segments only
        self._print_category_table(results['no_switch'], "SEGMENTS WITHOUT SWITCHES ONLY", include_mode_accuracy=True)
        self._print_winner_summary(results)
        # 4. Post-processing impact analysis
        self._print_pp_impact_analysis(results)

        # 5. Summary and rankings
        self._print_summary_rankings(results)
        # if hasattr(self, 'switch_diagnostics'):  # not strictly needed; safer if called elsewhere
        self._print_switch_diagnostics(switch_diagnostics)

    def _print_category_table(self, category_results: Dict, title: str, include_mode_accuracy: bool = False):
        """Print results table with models as rows and metrics as columns, marking best performers"""
        print(f"\n{'=' * 200}")
        print(f"{title} (N = {category_results[list(category_results.keys())[0]]['segment_count']} segments)")
        print(f"{'=' * 200}")

        # Define metrics to show
        metrics_to_show = [
            ('F-β(2)', 'proximity_fbeta2'),
            ('Precision', 'proximity_precision'),
            ('Recall', 'proximity_recall'),
            ('F1', 'proximity_f1'),
        ]

        if include_mode_accuracy:
            metrics_to_show.append(('Mode Acc', 'mode_accuracy'))

        metrics_to_show.extend([
            ('True SW', 'true_switches'),
            ('Pred SW', 'pred_switches'),
            ('Matches', 'total_matches'),
        ])

        # Find best value for each metric
        best_values = {}
        for display_name, key in metrics_to_show:
            if key in ['true_switches', 'pred_switches', 'total_matches']:
                # For count metrics, we don't mark "best" as they're descriptive
                best_values[key] = None
            else:
                # For performance metrics, find the max
                values = [category_results[model].get(key, 0) for model in category_results]
                best_values[key] = max(values) if values else 0

        # Column widths
        model_col_width = 34
        metric_col_width = 11

        # Print header
        header = f"{'Model':<{model_col_width}}"
        for display_name, _ in metrics_to_show:
            header += f"{display_name:>{metric_col_width}}"
        print(header)
        print("-" * len(header))

        # Print each model's results
        for model_name in self.config.model_names:
            if model_name not in category_results:
                continue

            row = f"{model_name:<{model_col_width}}"

            for display_name, key in metrics_to_show:
                val = category_results[model_name].get(key, 0)

                # Format value
                if key in ['true_switches', 'pred_switches', 'total_matches']:
                    val_str = f"{int(val):>{metric_col_width - 1}}"
                else:
                    val_str = f"{val:>{metric_col_width - 1}.3f}"

                # Mark if best (with asterisk)
                if best_values.get(key) is not None and val == best_values[key] and val > 0:
                    val_str += "*"
                else:
                    val_str += " "

                row += val_str

            print(row)

        # Print legend
        print("\n* = Best performer for this metric")

    def _print_pp_impact_analysis(self, results: Dict):
        """Analyze post-processing impact across all categories"""
        print("\n" + "=" * 200)
        print("POST-PROCESSING IMPACT ANALYSIS")
        print("=" * 200)

        model_pairs = [
            ('mBERT', 'mBERT+PP'),
            ('Simple-4Class', 'Simple-4Class+PP'),  # ADD
            ('Simple-3Class', 'Simple-3Class+PP'),  # ADD
            ('mBERT-3Class-_ADDITIVE_LOSS', 'mBERT-3Class-_ADDITIVE_LOSS+PP'),
            ('XLM-R_WEIGHT_LOSS', 'XLM-R_WEIGHT_LOSS+PP'),
            ('ALTO_MUL', 'ALTO_MUL+PP'),
            ('ALTO_MUL_SEGMENT_REWARD', 'ALTO_MUL_SEGMENT_REWARD+PP'),
            ('ALTO_ADDITIVE', 'ALTO_ADDITIVE+PP'),
            ('ALTO_ADDITIVE_SEGMENT_REWARD', 'ALTO_ADDITIVE_SEGMENT_REWARD+PP'),
            ('ALTO-4Class', 'ALTO-4Class+PP'),  # ADD THIS LINE
            ('ALTO-3Class_ADDITIVE_LOSS', 'ALTO-3Class_ADDITIVE_LOSS+PP'),  # ADD THIS LINE
            ('CINO_ADDITIVE_LOSS', 'CINO_ADDITIVE_LOSS+PP'),
            ('Tibetan-RoBERTa_ADDITIVE_LOSS', 'Tibetan-RoBERTa_ADDITIVE_LOSS+PP'),
        ]

        # Check if CRF is available
        # if 'CRF-Model' in results['all']:
        #     model_pairs.append(('CRF-Model', 'CRF-Model+PP'))

        for category, cat_name in [('all', 'ALL SEGMENTS'),
                                   ('switch', 'SWITCH SEGMENTS'),
                                   ('no_switch', 'NON-SWITCH SEGMENTS')]:

            print(f"\n{cat_name}:")
            print(f"{'Model':<20} {'Baseline':<12} {'+ PP':<12} {'Δ F-beta':<12} {'Δ Precision':<12} {'Δ Recall':<12}")
            print("-" * 90)

            for base, pp in model_pairs:
                if base in results[category] and pp in results[category]:
                    base_f = results[category][base]['proximity_fbeta2']
                    pp_f = results[category][pp]['proximity_fbeta2']

                    base_p = results[category][base]['proximity_precision']
                    pp_p = results[category][pp]['proximity_precision']

                    base_r = results[category][base]['proximity_recall']
                    pp_r = results[category][pp]['proximity_recall']

                    print(f"{base:<20} {base_f:<12.3f} {pp_f:<12.3f} {pp_f - base_f:+<12.3f} "
                          f"{pp_p - base_p:+<12.3f} {pp_r - base_r:+<12.3f}")

    def _print_switch_diagnostics(self, switch_diag: Dict[str, Dict]):
        print("\n" + "=" * 160)
        print("SWITCH SPACING & REPETITION DIAGNOSTICS (predictions; ALL segments)")
        print("=" * 160)

        models = [m for m in self.config.model_names if m in switch_diag]
        col_w = 12
        header = (
            f"{'Model':<22}"
            f"{'Pairs':>{col_w}}"
            f"{'MeanGap':>{col_w}}"
            f"{'MedGap':>{col_w}}"
            f"{'Min':>{col_w}}"
            f"{'Max':>{col_w}}"
            f"{'Adj%':>{col_w}}"  # gaps==0
            f"{'1Gap%':>{col_w}}"  # gaps==1
            f"{'Same%':>{col_w}}"  # consecutive same type
            f"{'MaxRun':>{col_w}}"  # longest same-type run
        )
        print(header)
        print("-" * len(header))

        for m in models:
            d = switch_diag[m]
            print(
                f"{m:<22}"
                f"{d['pairs']:>{col_w}d}"
                f"{d['mean_gap']:>{col_w}.2f}"
                f"{d['median_gap']:>{col_w}.2f}"
                f"{d['min_gap']:>{col_w}d}"
                f"{d['max_gap']:>{col_w}d}"
                f"{(100 * d['pct_adjacent']):>{col_w}.1f}"
                f"{(100 * d['pct_single_gap']):>{col_w}.1f}"
                f"{(100 * d['pct_same_type_consec']):>{col_w}.1f}"
                f"{d['max_same_type_run']:>{col_w}d}"
            )

    def _print_summary_rankings(self, results: Dict):
        """Print summary and rankings"""
        print("\n" + "=" * 200)
        print("OVERALL MODEL RANKINGS (by F-beta(2) on ALL segments)")
        print("=" * 200)

        ranked = sorted(results['all'].items(),
                        key=lambda x: x[1]['proximity_fbeta2'],
                        reverse=True)

        print(f"\n{'Rank':<6} {'Model':<20} {'All F-β(2)':<12} {'Switch F-β(2)':<15} {'NoSwitch Mode Acc':<20}")
        print("-" * 75)

        for rank, (name, metrics) in enumerate(ranked, 1):
            all_fbeta = metrics['proximity_fbeta2']
            switch_fbeta = results['switch'][name]['proximity_fbeta2'] if name in results['switch'] else 0
            no_switch_acc = results['no_switch'][name]['mode_accuracy'] if name in results['no_switch'] else 0

            print(f"{rank:<6} {name:<20} {all_fbeta:<12.3f} {switch_fbeta:<15.3f} {no_switch_acc:<20.3f}")

        # Best in each category
        print("\n" + "=" * 200)
        print("CATEGORY WINNERS")
        print("=" * 200)

        # Best overall
        best_overall = ranked[0]
        print(f"🏆 Best Overall: {best_overall[0]} (F-β(2) = {best_overall[1]['proximity_fbeta2']:.3f})")

        # Best on switch segments
        switch_ranked = sorted(results['switch'].items(),
                               key=lambda x: x[1]['proximity_fbeta2'],
                               reverse=True)
        if switch_ranked:
            print(
                f"🎯 Best on Switch Segments: {switch_ranked[0][0]} (F-β(2) = {switch_ranked[0][1]['proximity_fbeta2']:.3f})")

        # Best on non-switch segments
        no_switch_ranked = sorted(results['no_switch'].items(),
                                  key=lambda x: x[1]['mode_accuracy'],
                                  reverse=True)
        if no_switch_ranked:
            print(
                f"📊 Best on Non-Switch Segments: {no_switch_ranked[0][0]} (Mode Acc = {no_switch_ranked[0][1]['mode_accuracy']:.3f})")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_comprehensive_evaluation(config: EvaluationConfig = None):
    """Main comprehensive evaluation pipeline"""
    if config is None:
        config = EvaluationConfig()

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(config.test_file)
    print(f"Test set: {len(test_df)} segments")

    # Count segments with/without switches
    segments_with_switches = 0
    segments_without_switches = 0
    total_switches = 0

    for idx in range(len(test_df)):
        labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
        switches = sum(1 for l in labels if l in [2, 3])
        total_switches += switches
        if switches > 0:
            segments_with_switches += 1
        else:
            segments_without_switches += 1

    print(f"  Segments with switches: {segments_with_switches}")
    print(f"  Segments without switches: {segments_without_switches}")
    print(f"  Total switches in dataset: {total_switches}")
    print(f"  Average switches per segment: {total_switches / len(test_df):.2f}")

    # Load models
    model_manager = ModelManager(config).load_all_models()

    # Run evaluation
    evaluator = ComprehensiveEvaluator(config)
    results, segment_metrics, switch_diagnostics = evaluator.evaluate_all_models(test_df, model_manager)




    # Print results
    reporter = ComprehensiveReporter(config)
    reporter.print_all_results(results, switch_diagnostics)

    return results, segment_metrics

def _extract_switch_positions_and_types(labels):
    """Return (positions, types) for labels that are switches {2,3}."""
    pos, typ = [], []
    for i, l in enumerate(labels):
        if l in (2, 3):
            pos.append(i)
            typ.append(l)
    return pos, typ


def switch_sequence_diagnostics(pred_labels):
    """
    Diagnostics over predicted switches only.
    - gaps: list of tokens-between-switches (j - i - 1)
    - adjacent_pairs: gaps == 0
    - single_gap_pairs: gaps == 1
    - same_type_pairs: consecutive switch types equal (e.g., 3,3 or 2,2)
    - max_same_type_run: longest run of same switch type in the switch stream
    """
    idxs, types = _extract_switch_positions_and_types(pred_labels)
    n = len(idxs)
    diag = {
        "pair_count": max(0, n - 1),
        "gaps": [],
        "adjacent_pairs": 0,
        "single_gap_pairs": 0,
        "same_type_pairs": 0,
        "max_same_type_run": 0,
    }
    if n <= 1:
        return diag

    # gaps & immediate issues
    for k in range(n - 1):
        gap_between = idxs[k + 1] - idxs[k] - 1  # tokens between the two switches
        diag["gaps"].append(gap_between)
        if gap_between == 0:
            diag["adjacent_pairs"] += 1
        if gap_between == 1:
            diag["single_gap_pairs"] += 1

    # same-type consecutive checks in switch-only stream
    run = 1
    for k in range(n - 1):
        if types[k] == types[k + 1]:
            diag["same_type_pairs"] += 1
            run += 1
        else:
            diag["max_same_type_run"] = max(diag["max_same_type_run"], run)
            run = 1
    diag["max_same_type_run"] = max(diag["max_same_type_run"], run)

    return diag


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def verify_remapping_preserves_switches():
    """Verify remapping preserves switches and tracks mode correctly."""
    print("=" * 70)
    print("VERIFYING REMAPPING FUNCTION")
    print("=" * 70)

    # Test: Simple switch sequence
    preds_3class = [0, 0, 2, 0, 0, 1, 0, 0]
    #              Auto Auto →Allo Allo Allo →Auto Auto Auto

    remapped = remap_3class_to_4class(preds_3class)
    expected = [0, 0, 3, 1, 1, 2, 0, 0]

    print(f"\n3-class:  {preds_3class}")
    print(f"Remapped: {remapped}")
    print(f"Expected: {expected}")

    # Verify switches preserved
    switches_3 = [i for i, p in enumerate(preds_3class) if p in [1, 2]]
    switches_r = [i for i, p in enumerate(remapped) if p in [2, 3]]

    assert remapped == expected, "Remapping failed!"
    assert switches_3 == switches_r, "Switches not preserved!"

    print("\n✅ ALL TESTS PASSED - REMAPPING IS CORRECT!")
    print("=" * 70)


# Run before evaluation
if __name__ == "__main__":
    verify_remapping_preserves_switches()

    config = EvaluationConfig()

    # Update paths to your trained simple models
    config.simple_4class_model_path = './alloauto-segmentation-training/benchmark_models_standard/simple_mBert_vanilla_benchmark_4_class_NER_23_10_fixed/final_model'
    # config.simple_4class_model_path = './alloauto-segmentation-training/benchmark_models/simple_mBert_vanilla_benchmark_3_class_NER/final_model'
    config.simple_3class_model_path = './alloauto-segmentation-training/benchmark_models_standard/simple_mBert_vanilla_benchmark_3_class_NER_23_10/final_model'
    # config.simple_3class_model_path = './alloauto-segmentation-training/benchmark_models/simple_mBert_vanilla_benchmark_3_class_NER/final_model'

    results, segment_metrics = run_comprehensive_evaluation()

