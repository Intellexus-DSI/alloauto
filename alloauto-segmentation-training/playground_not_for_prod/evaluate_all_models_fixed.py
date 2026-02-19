"""
Comprehensive Code-Switching Evaluation Framework
Evaluates models on ALL segments, switch segments, and non-switch segments
Includes fixed post-processing for mBERT, XLM-R, and ALTO
UPDATED: Supports both 3-class and 4-class models with proper data preprocessing
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
                                min_tokens_between_switches: int = 2,
                                num_classes: int = 4) -> Union[np.ndarray, List[int]]:
    """
    Apply post-processing rules to enforce logical consistency in predictions.

    Rules enforced (for 4-class):
    1. No adjacent switches (switches cannot be back-to-back)
    2. Switch→Allo (label 3) must follow Non-switch Auto (label 0)
    3. Switch→Auto (label 2) must follow Non-switch Allo (label 1)
    4. Minimum tokens between switches (default: 2)
       - After Switch→Allo: at least 2 Allo tokens before next switch
       - After Switch→Auto: at least 2 Auto tokens before next switch

    For 3-class models:
    - Label 0: Non-switch (either Auto or Allo)
    - Label 1: Switch→Auto
    - Label 2: Switch→Allo

    Label mapping (4-class):
        0: Non-switch Auto
        1: Non-switch Allo
        2: Switch→Auto
        3: Switch→Allo

    Args:
        predictions: Array or list of predicted labels
        min_tokens_between_switches: Minimum number of tokens required between switches
        num_classes: Number of classes (3 or 4)

    Returns:
        Corrected predictions in same format as input
    """
    if len(predictions) == 0:
        return predictions

    # Convert to list for easier manipulation
    is_numpy = isinstance(predictions, np.ndarray)
    preds = predictions.tolist() if is_numpy else list(predictions)

    if num_classes == 3:
        # For 3-class: simpler rules
        # 0: Non-switch, 1: Switch→Auto, 2: Switch→Allo
        corrected = []
        tokens_since_last_switch = min_tokens_between_switches

        for i, pred in enumerate(preds):
            if pred == -100:
                corrected.append(pred)
                continue

            is_switch = pred in [1, 2]

            if is_switch:
                # Check minimum spacing
                if tokens_since_last_switch < min_tokens_between_switches:
                    corrected.append(0)  # Convert to non-switch
                    tokens_since_last_switch += 1
                else:
                    corrected.append(pred)
                    tokens_since_last_switch = 0
            else:
                corrected.append(pred)
                tokens_since_last_switch += 1

        return np.array(corrected) if is_numpy else corrected

    # 4-class logic (original)
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


def verify_post_processing_rules(predictions: Union[np.ndarray, List[int]],
                                 num_classes: int = 4) -> dict:
    """
    Verify that predictions follow all post-processing rules.
    Useful for debugging and validation.

    Args:
        predictions: Array or list of predicted labels
        num_classes: Number of classes (3 or 4)

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

    if num_classes == 3:
        # For 3-class: simpler validation
        tokens_since_last_switch = 0
        min_spacing = 2

        for i, pred in enumerate(preds):
            if pred == -100:
                continue

            is_switch = pred in [1, 2]

            if is_switch:
                violations['total_switches'] += 1

                if tokens_since_last_switch == 0:
                    violations['adjacent_switches'].append(i)

                if tokens_since_last_switch < min_spacing:
                    violations['insufficient_spacing'].append({
                        'position': i,
                        'tokens_since_last': tokens_since_last_switch
                    })

                tokens_since_last_switch = 0
            else:
                tokens_since_last_switch += 1

        return violations

    # 4-class validation (original)
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


def convert_labels_4class_to_3class(labels: List[int]) -> List[int]:
    """
    Convert 4-class labels to 3-class labels.

    4-class mapping:
        0: Non-switch Auto
        1: Non-switch Allo
        2: Switch→Auto
        3: Switch→Allo

    3-class mapping:
        0: Non-switch (merges Auto and Allo)
        1: Switch→Auto
        2: Switch→Allo

    Args:
        labels: List of 4-class labels

    Returns:
        List of 3-class labels
    """
    converted = []
    for label in labels:
        if label == -100:  # Padding
            converted.append(-100)
        elif label in [0, 1]:  # Non-switch Auto or Allo → Non-switch
            converted.append(0)
        elif label == 2:  # Switch→Auto stays as 1
            converted.append(1)
        elif label == 3:  # Switch→Allo becomes 2
            converted.append(2)
        else:
            raise ValueError(f"Unexpected label value: {label}")
    return converted


def convert_labels_3class_to_4class_for_eval(labels_3class: List[int],
                                             predictions_3class: List[int],
                                             original_4class_labels: List[int] = None) -> Tuple[List[int], List[int]]:
    """
    Convert 3-class predictions back to 4-class format for fair comparison.
    This requires context about what the non-switch tokens actually are.

    If original_4class_labels is provided, we can do proper conversion.
    Otherwise, we make assumptions based on context.

    Args:
        labels_3class: 3-class ground truth labels
        predictions_3class: 3-class predictions
        original_4class_labels: Optional original 4-class labels for reference

    Returns:
        Tuple of (converted_labels_4class, converted_predictions_4class)
    """
    if original_4class_labels is not None:
        # Use original labels to determine Auto vs Allo for non-switch tokens
        converted_preds = []
        converted_labels = []

        for i, (pred_3, label_3, orig_4) in enumerate(zip(predictions_3class, labels_3class, original_4class_labels)):
            if pred_3 == 0:  # Non-switch prediction
                # Use context from original to determine if Auto (0) or Allo (1)
                if orig_4 in [0, 1]:
                    converted_preds.append(orig_4)  # Keep original mode
                else:
                    # Fallback: look at previous prediction or default to Auto
                    if i > 0 and converted_preds[-1] in [0, 1]:
                        converted_preds.append(converted_preds[-1])
                    else:
                        converted_preds.append(0)  # Default to Auto
            elif pred_3 == 1:  # Switch→Auto
                converted_preds.append(2)
            elif pred_3 == 2:  # Switch→Allo
                converted_preds.append(3)
            elif pred_3 == -100:
                converted_preds.append(-100)

            # Convert labels similarly
            if label_3 == 0:
                converted_labels.append(orig_4 if orig_4 in [0, 1] else 0)
            elif label_3 == 1:
                converted_labels.append(2)
            elif label_3 == 2:
                converted_labels.append(3)
            elif label_3 == -100:
                converted_labels.append(-100)

        return converted_labels, converted_preds
    else:
        # Without original labels, use sequential logic
        converted_preds = []
        current_mode = 0  # Start with Auto

        for pred_3 in predictions_3class:
            if pred_3 == 0:  # Non-switch
                converted_preds.append(current_mode)
            elif pred_3 == 1:  # Switch→Auto
                converted_preds.append(2)
                current_mode = 0
            elif pred_3 == 2:  # Switch→Allo
                converted_preds.append(3)
                current_mode = 1
            elif pred_3 == -100:
                converted_preds.append(-100)

        # For labels, do similar conversion
        converted_labels = []
        current_mode = 0

        for label_3 in labels_3class:
            if label_3 == 0:
                converted_labels.append(current_mode)
            elif label_3 == 1:
                converted_labels.append(2)
                current_mode = 0
            elif label_3 == 2:
                converted_labels.append(3)
                current_mode = 1
            elif label_3 == -100:
                converted_labels.append(-100)

        return converted_labels, converted_preds


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def example_batch_inference_with_post_processing(model, tokenizer, texts: List[str],
                                                 batch_size: int = 8,
                                                 num_classes: int = 4) -> List[List[int]]:
    """
    Example showing how to apply post-processing during batch inference.

    Args:
        model: Your trained model
        tokenizer: Corresponding tokenizer
        texts: List of input texts
        batch_size: Batch size for inference
        num_classes: Number of classes (3 or 4)

    Returns:
        List of corrected prediction sequences
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_corrected_predictions = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            # Apply post-processing to each sequence
            for j, pred_seq in enumerate(predictions):
                # Get actual length (excluding padding)
                actual_length = attention_mask[j].sum().item()

                # Extract non-padded predictions
                valid_preds = pred_seq[:actual_length].tolist()

                # Apply post-processing rules
                corrected_preds = apply_post_processing_rules(valid_preds, num_classes=num_classes)

                all_corrected_predictions.append(corrected_preds)

    return all_corrected_predictions


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation"""
    # Test file
    test_file: str = 'dataset/preprocessed_augmented/test_segments_original.csv'

    # UPDATED: Model paths for all 6 models
    mBert_alto_loss_MUL_SEG_3_classes: str = './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mbert_cased_ALTO_MUL_SEG_arch_3class_23_10/final_model'
    mBert_alto_loss_MUL_SEG_4_classes: str = './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mbert_cased_ALTO_arch_23_10/final_model'
    mBert_alto_loss_ADDITIVE_SEG_3_classes: str = './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mBERT_additive_seg_loss_3class_23_10/final_model'
    mBert_alto_loss_ADDITIVE_SEG_4_classes: str = './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mBERT_vanilla_model_additive_loss_seg_23_10/final_model'
    mBert_simple_3_class: str = './alloauto-segmentation-training/benchmark_models_standard/simple_mBert_vanilla_benchmark_3_class_NER_23_10/final_model'
    mBert_simple_4_class: str = './alloauto-segmentation-training/benchmark_models_standard/simple_mBert_vanilla_benchmark_4_class_NER_23_10/final_model'

    # Model names for reporting
    model_names: List[str] = None

    # UPDATED: Track which models use 3 classes vs 4 classes
    three_class_models: List[str] = None
    four_class_models: List[str] = None

    # Batch size
    batch_size: int = 16

    # Post-processing
    apply_postprocessing: bool = True
    min_tokens_between_switches: int = 2

    # Proximity threshold
    proximity_threshold: int = 5

    def __post_init__(self):
        if self.model_names is None:
            self.model_names = [
                'mBert_alto_MUL_SEG_3cls',
                'mBert_alto_MUL_SEG_4cls',
                'mBert_alto_ADD_SEG_3cls',
                'mBert_alto_ADD_SEG_4cls',
                'mBert_simple_3cls',
                'mBert_simple_4cls'
            ]

        if self.three_class_models is None:
            self.three_class_models = [
                'mBert_alto_MUL_SEG_3cls',
                'mBert_alto_ADD_SEG_3cls',
                'mBert_simple_3cls'
            ]

        if self.four_class_models is None:
            self.four_class_models = [
                'mBert_alto_MUL_SEG_4cls',
                'mBert_alto_ADD_SEG_4cls',
                'mBert_simple_4cls'
            ]

    def is_three_class_model(self, model_name: str) -> bool:
        """Check if a model uses 3 classes"""
        return model_name in self.three_class_models

    def get_num_classes(self, model_name: str) -> int:
        """Get number of classes for a model"""
        return 3 if self.is_three_class_model(model_name) else 4

    def get_model_path(self, model_name: str) -> str:
        """Get the path for a given model name"""
        path_mapping = {
            'mBert_alto_MUL_SEG_3cls': self.mBert_alto_loss_MUL_SEG_3_classes,
            'mBert_alto_MUL_SEG_4cls': self.mBert_alto_loss_MUL_SEG_4_classes,
            'mBert_alto_ADD_SEG_3cls': self.mBert_alto_loss_ADDITIVE_SEG_3_classes,
            'mBert_alto_ADD_SEG_4cls': self.mBert_alto_loss_ADDITIVE_SEG_4_classes,
            'mBert_simple_3cls': self.mBert_simple_3_class,
            'mBert_simple_4cls': self.mBert_simple_4_class
        }
        return path_mapping.get(model_name, '')


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Manages loading and inference for all model types"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_all_models(self):
        """Load all models specified in config"""
        print("\nLoading models...")

        for model_name in self.config.model_names:
            model_path = self.config.get_model_path(model_name)
            num_classes = self.config.get_num_classes(model_name)

            if not model_path or not Path(model_path).exists():
                print(f"⚠️  Skipping {model_name}: path not found - {model_path}")
                continue

            try:
                print(f"  Loading {model_name} ({num_classes} classes)...")

                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForTokenClassification.from_pretrained(model_path)
                model.to(self.device)
                model.eval()

                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer

                print(f"    ✓ Loaded successfully")

            except Exception as e:
                print(f"    ✗ Error loading {model_name}: {e}")

        print(f"\n✓ Loaded {len(self.models)} models successfully")
        return self

    def predict_batch(self, model_name: str, texts: List[str]) -> List[List[int]]:
        """
        Get predictions for a batch of texts

        Args:
            model_name: Name of the model to use
            texts: List of input texts

        Returns:
            List of prediction sequences (one per text)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        num_classes = self.config.get_num_classes(model_name)

        all_predictions = []

        with torch.no_grad():
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]

                # Tokenize
                encodings = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # Move to device
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)

                # Get predictions
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

                # Process each sequence
                for j, pred_seq in enumerate(predictions):
                    # Get actual length (excluding padding)
                    actual_length = attention_mask[j].sum().item()

                    # Extract non-padded predictions
                    valid_preds = pred_seq[:actual_length].tolist()

                    # Apply post-processing if enabled
                    if self.config.apply_postprocessing:
                        valid_preds = apply_post_processing_rules(
                            valid_preds,
                            min_tokens_between_switches=self.config.min_tokens_between_switches,
                            num_classes=num_classes
                        )

                    all_predictions.append(valid_preds)

        return all_predictions


# ============================================================================
# METRICS & EVALUATION
# ============================================================================

def proximity_metrics(true_positions: List[int], pred_positions: List[int],
                      true_labels: List[int], pred_labels: List[int],
                      threshold: int = 5) -> Dict[str, float]:
    """
    Calculate proximity-based metrics for switch detection.
    A prediction is correct if it's within threshold tokens of a true switch
    and has the correct type (auto/allo).

    EXACT MATCH of original evaluate_switch_detection_with_proximity logic
    """
    if len(true_positions) == 0:
        return {
            'proximity_precision': 0.0 if len(pred_positions) > 0 else 1.0,
            'proximity_recall': 0.0,
            'proximity_fbeta2': 0.0,
            'exact_matches': 0,
            'proximity_matches': 0,
            'total_matches': 0,
            'total_true_switches': 0,
            'total_pred_switches': len(pred_positions)
        }

    if len(pred_positions) == 0:
        return {
            'proximity_precision': 0.0,
            'proximity_recall': 0.0,
            'proximity_fbeta2': 0.0,
            'exact_matches': 0,
            'proximity_matches': 0,
            'total_matches': 0,
            'total_true_switches': len(true_positions),
            'total_pred_switches': 0
        }

    # Track matches
    true_matched = set()
    pred_matched = set()
    exact_matches = 0
    proximity_matches = 0

    # For each prediction, find closest true switch
    for pred_idx, pred_pos in enumerate(pred_positions):
        pred_type = pred_labels[pred_idx]

        best_distance = float('inf')
        best_true_idx = None

        for true_idx, true_pos in enumerate(true_positions):
            if true_idx in true_matched:
                continue

            true_type = true_labels[true_idx]
            distance = abs(pred_pos - true_pos)

            # Match if within threshold AND same type
            if distance <= threshold and pred_type == true_type:
                if distance < best_distance:
                    best_distance = distance
                    best_true_idx = true_idx

        if best_true_idx is not None:
            true_matched.add(best_true_idx)
            pred_matched.add(pred_idx)
            if best_distance == 0:
                exact_matches += 1
            else:
                proximity_matches += 1

    # Calculate metrics
    total_true_switches = len(true_positions)
    total_pred_switches = len(pred_positions)
    total_matches = exact_matches + proximity_matches

    proximity_precision = total_matches / total_pred_switches if total_pred_switches > 0 else 0.0
    proximity_recall = total_matches / total_true_switches if total_true_switches > 0 else 0.0

    # F-beta with beta=2 (emphasizes recall)
    beta = 2
    proximity_fbeta2 = ((1 + beta ** 2) * proximity_precision * proximity_recall) / (
                (beta ** 2 * proximity_precision) + proximity_recall) if (
                                                                                     proximity_precision + proximity_recall) > 0 else 0.0

    return {
        'proximity_precision': proximity_precision,
        'proximity_recall': proximity_recall,
        'proximity_fbeta2': proximity_fbeta2,
        'exact_matches': exact_matches,
        'proximity_matches': proximity_matches,
        'total_matches': total_matches,
        'total_true_switches': total_true_switches,
        'total_pred_switches': total_pred_switches
    }


def mode_accuracy(true_labels: List[int], pred_labels: List[int], num_classes: int = 4) -> float:
    """
    Calculate mode accuracy for non-switch segments.
    For 4-class: distinguishes between Auto (0) and Allo (1)
    For 3-class: all non-switches are class 0, so mode accuracy is always 1.0
    """
    if num_classes == 3:
        # For 3-class, we don't distinguish between Auto/Allo non-switches
        # So mode accuracy is essentially 100% by definition if labels match
        non_switch_true = [l for l in true_labels if l == 0]
        non_switch_pred = [p for i, p in enumerate(pred_labels) if true_labels[i] == 0]

        if len(non_switch_true) == 0:
            return 1.0

        correct = sum(1 for t, p in zip(non_switch_true, non_switch_pred) if t == p)
        return correct / len(non_switch_true)
    else:
        # For 4-class: compare Auto (0) vs Allo (1) for non-switch tokens
        non_switch_true = [l for l in true_labels if l in [0, 1]]
        non_switch_pred = [p for i, p in enumerate(pred_labels) if true_labels[i] in [0, 1]]

        if len(non_switch_true) == 0:
            return 1.0

        correct = sum(1 for t, p in zip(non_switch_true, non_switch_pred) if t == p)
        return correct / len(non_switch_true)


# ============================================================================
# COMPREHENSIVE EVALUATOR
# ============================================================================

class ComprehensiveEvaluator:
    """Evaluates models on all segments, switch segments, and non-switch segments"""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def prepare_data_for_model(self, test_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Prepare test data for a specific model.
        For 3-class models, convert labels from 4-class to 3-class.

        Args:
            test_df: Original test dataframe (with 4-class labels)
            model_name: Name of the model

        Returns:
            Prepared dataframe (potentially with converted labels)
        """
        if not self.config.is_three_class_model(model_name):
            # 4-class model: return original data
            return test_df.copy()

        # 3-class model: convert labels
        print(f"  Converting labels to 3-class format for {model_name}...")

        df_3class = test_df.copy()
        converted_labels = []

        for idx in range(len(df_3class)):
            labels_4class = [int(l) for l in df_3class.iloc[idx]['labels'].split(',')]
            labels_3class = convert_labels_4class_to_3class(labels_4class)
            converted_labels.append(','.join(map(str, labels_3class)))

        df_3class['labels'] = converted_labels
        df_3class['original_4class_labels'] = test_df['labels']  # Keep original for reference

        return df_3class

    def evaluate_all_models(self, test_df: pd.DataFrame, model_manager: ModelManager) -> Tuple[Dict, Dict, Dict]:
        """
        Evaluate all models comprehensively.

        Returns:
            Tuple of (results, segment_metrics, switch_diagnostics)
        """
        results = {
            'all': {},
            'switch': {},
            'no_switch': {}
        }

        segment_metrics = {}
        switch_diagnostics = {}

        for model_name in self.config.model_names:
            if model_name not in model_manager.models:
                print(f"⚠️  Skipping {model_name}: not loaded")
                continue

            print(f"\n{'=' * 80}")
            print(f"Evaluating {model_name}")
            print(f"{'=' * 80}")

            # Prepare data for this model (convert to 3-class if needed)
            model_test_df = self.prepare_data_for_model(test_df, model_name)
            num_classes = self.config.get_num_classes(model_name)

            # Get predictions
            print("  Running inference...")
            texts = model_test_df['text'].tolist()
            predictions = model_manager.predict_batch(model_name, texts)

            # Evaluate on all segments
            print("  Evaluating on ALL segments...")
            all_metrics = self._evaluate_on_all_segments(
                model_test_df, predictions, num_classes
            )
            results['all'][model_name] = all_metrics

            # Evaluate on switch segments only
            print("  Evaluating on SWITCH segments...")
            switch_metrics = self._evaluate_on_switch_segments(
                model_test_df, predictions, num_classes
            )
            results['switch'][model_name] = switch_metrics

            # Evaluate on non-switch segments
            print("  Evaluating on NON-SWITCH segments...")
            no_switch_metrics = self._evaluate_on_non_switch_segments(
                model_test_df, predictions, num_classes
            )
            results['no_switch'][model_name] = no_switch_metrics

            # Segment-level metrics
            segment_metrics[model_name] = self._compute_segment_metrics(
                model_test_df, predictions, num_classes
            )

            # Switch diagnostics
            print("  Computing switch diagnostics...")
            switch_diagnostics[model_name] = self._compute_switch_diagnostics(predictions)

        return results, segment_metrics, switch_diagnostics

    def _evaluate_on_all_segments(self, test_df: pd.DataFrame,
                                  predictions: List[List[int]],
                                  num_classes: int) -> Dict:
        """Evaluate on all segments"""
        all_true_labels = []
        all_pred_labels = []
        all_true_positions = []
        all_pred_positions = []
        all_true_types = []
        all_pred_types = []

        offset = 0

        for idx in range(len(test_df)):
            true_labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
            pred_labels = predictions[idx]

            # Ensure same length
            min_len = min(len(true_labels), len(pred_labels))
            true_labels = true_labels[:min_len]
            pred_labels = pred_labels[:min_len]

            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

            # Extract switch positions and types
            if num_classes == 4:
                true_switches = [(i + offset, l) for i, l in enumerate(true_labels) if l in [2, 3]]
                pred_switches = [(i + offset, l) for i, l in enumerate(pred_labels) if l in [2, 3]]
            else:  # 3-class
                true_switches = [(i + offset, l) for i, l in enumerate(true_labels) if l in [1, 2]]
                pred_switches = [(i + offset, l) for i, l in enumerate(pred_labels) if l in [1, 2]]

            if true_switches:
                all_true_positions.extend([pos for pos, _ in true_switches])
                all_true_types.extend([typ for _, typ in true_switches])

            if pred_switches:
                all_pred_positions.extend([pos for pos, _ in pred_switches])
                all_pred_types.extend([typ for _, typ in pred_switches])

            offset += len(true_labels)

        # Proximity metrics for switches
        prox_metrics = proximity_metrics(
            all_true_positions, all_pred_positions,
            all_true_types, all_pred_types,
            threshold=self.config.proximity_threshold
        )

        # Mode accuracy for non-switches
        mode_acc = mode_accuracy(all_true_labels, all_pred_labels, num_classes)

        # Overall accuracy
        accuracy = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == p) / len(all_true_labels)

        return {
            **prox_metrics,
            'mode_accuracy': mode_acc,
            'overall_accuracy': accuracy,
            'total_true_switches': len(all_true_positions),
            'total_pred_switches': len(all_pred_positions)
        }

    def _evaluate_on_switch_segments(self, test_df: pd.DataFrame,
                                     predictions: List[List[int]],
                                     num_classes: int) -> Dict:
        """Evaluate only on segments that contain switches"""
        switch_segment_indices = []

        # Identify segments with switches
        for idx in range(len(test_df)):
            labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
            if num_classes == 4:
                has_switch = any(l in [2, 3] for l in labels)
            else:  # 3-class
                has_switch = any(l in [1, 2] for l in labels)

            if has_switch:
                switch_segment_indices.append(idx)

        if not switch_segment_indices:
            return {
                'proximity_precision': 0.0,
                'proximity_recall': 0.0,
                'proximity_fbeta2': 0.0,
                'segment_count': 0
            }

        # Evaluate on switch segments
        all_true_labels = []
        all_pred_labels = []
        all_true_positions = []
        all_pred_positions = []
        all_true_types = []
        all_pred_types = []

        offset = 0

        for idx in switch_segment_indices:
            true_labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
            pred_labels = predictions[idx]

            min_len = min(len(true_labels), len(pred_labels))
            true_labels = true_labels[:min_len]
            pred_labels = pred_labels[:min_len]

            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

            # Extract switches
            if num_classes == 4:
                true_switches = [(i + offset, l) for i, l in enumerate(true_labels) if l in [2, 3]]
                pred_switches = [(i + offset, l) for i, l in enumerate(pred_labels) if l in [2, 3]]
            else:
                true_switches = [(i + offset, l) for i, l in enumerate(true_labels) if l in [1, 2]]
                pred_switches = [(i + offset, l) for i, l in enumerate(pred_labels) if l in [1, 2]]

            if true_switches:
                all_true_positions.extend([pos for pos, _ in true_switches])
                all_true_types.extend([typ for _, typ in true_switches])

            if pred_switches:
                all_pred_positions.extend([pos for pos, _ in pred_switches])
                all_pred_types.extend([typ for _, typ in pred_switches])

            offset += len(true_labels)

        prox_metrics = proximity_metrics(
            all_true_positions, all_pred_positions,
            all_true_types, all_pred_types,
            threshold=self.config.proximity_threshold
        )

        prox_metrics['segment_count'] = len(switch_segment_indices)
        return prox_metrics

    def _evaluate_on_non_switch_segments(self, test_df: pd.DataFrame,
                                         predictions: List[List[int]],
                                         num_classes: int) -> Dict:
        """Evaluate only on segments without switches"""
        no_switch_segment_indices = []

        # Identify segments without switches
        for idx in range(len(test_df)):
            labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
            if num_classes == 4:
                has_switch = any(l in [2, 3] for l in labels)
            else:
                has_switch = any(l in [1, 2] for l in labels)

            if not has_switch:
                no_switch_segment_indices.append(idx)

        if not no_switch_segment_indices:
            return {
                'mode_accuracy': 0.0,
                'segment_count': 0
            }

        # Evaluate mode accuracy on non-switch segments
        all_true_labels = []
        all_pred_labels = []

        for idx in no_switch_segment_indices:
            true_labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
            pred_labels = predictions[idx]

            min_len = min(len(true_labels), len(pred_labels))
            true_labels = true_labels[:min_len]
            pred_labels = pred_labels[:min_len]

            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

        mode_acc = mode_accuracy(all_true_labels, all_pred_labels, num_classes)

        return {
            'mode_accuracy': mode_acc,
            'segment_count': len(no_switch_segment_indices),
            'overall_accuracy': sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == p) / len(
                all_true_labels) if all_true_labels else 0.0
        }

    def _compute_segment_metrics(self, test_df: pd.DataFrame,
                                 predictions: List[List[int]],
                                 num_classes: int) -> List[Dict]:
        """Compute per-segment metrics"""
        segment_metrics = []

        for idx in range(len(test_df)):
            true_labels = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
            pred_labels = predictions[idx]

            min_len = min(len(true_labels), len(pred_labels))
            true_labels = true_labels[:min_len]
            pred_labels = pred_labels[:min_len]

            # Extract switches
            if num_classes == 4:
                true_positions = [i for i, l in enumerate(true_labels) if l in [2, 3]]
                pred_positions = [i for i, l in enumerate(pred_labels) if l in [2, 3]]
                true_types = [l for l in true_labels if l in [2, 3]]
                pred_types = [l for l in pred_labels if l in [2, 3]]
            else:
                true_positions = [i for i, l in enumerate(true_labels) if l in [1, 2]]
                pred_positions = [i for i, l in enumerate(pred_labels) if l in [1, 2]]
                true_types = [l for l in true_labels if l in [1, 2]]
                pred_types = [l for l in pred_labels if l in [1, 2]]

            prox_metrics = proximity_metrics(
                true_positions, pred_positions,
                true_types, pred_types,
                threshold=self.config.proximity_threshold
            )

            mode_acc = mode_accuracy(true_labels, pred_labels, num_classes)

            segment_metrics.append({
                'segment_id': idx,
                'has_switches': len(true_positions) > 0,
                'num_true_switches': len(true_positions),
                'num_pred_switches': len(pred_positions),
                **prox_metrics,
                'mode_accuracy': mode_acc
            })

        return segment_metrics

    def _compute_switch_diagnostics(self, predictions: List[List[int]]) -> Dict:
        """Compute switch spacing and repetition diagnostics"""
        all_gaps = []
        all_same_type_pairs = 0
        all_adjacent = 0
        all_single_gap = 0
        all_pairs = 0
        max_run = 0

        for pred_labels in predictions:
            diag = switch_sequence_diagnostics(pred_labels)
            all_gaps.extend(diag['gaps'])
            all_same_type_pairs += diag['same_type_pairs']
            all_adjacent += diag['adjacent_pairs']
            all_single_gap += diag['single_gap_pairs']
            all_pairs += diag['pair_count']
            max_run = max(max_run, diag['max_same_type_run'])

        if not all_gaps:
            return {
                'pairs': 0,
                'mean_gap': 0.0,
                'median_gap': 0.0,
                'min_gap': 0,
                'max_gap': 0,
                'pct_adjacent': 0.0,
                'pct_single_gap': 0.0,
                'pct_same_type_consec': 0.0,
                'max_same_type_run': 0
            }

        return {
            'pairs': all_pairs,
            'mean_gap': np.mean(all_gaps),
            'median_gap': np.median(all_gaps),
            'min_gap': int(np.min(all_gaps)),
            'max_gap': int(np.max(all_gaps)),
            'pct_adjacent': all_adjacent / all_pairs if all_pairs > 0 else 0.0,
            'pct_single_gap': all_single_gap / all_pairs if all_pairs > 0 else 0.0,
            'pct_same_type_consec': all_same_type_pairs / all_pairs if all_pairs > 0 else 0.0,
            'max_same_type_run': max_run
        }


# ============================================================================
# REPORTER
# ============================================================================

class ComprehensiveReporter:
    """Prints comprehensive evaluation results"""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def print_all_results(self, results: Dict, switch_diag: Dict):
        """Print all evaluation results"""
        self._print_main_results(results)
        self._print_switch_diagnostics(switch_diag)
        self._print_summary_rankings(results)

    def _print_main_results(self, results: Dict):
        """Print main evaluation results"""
        categories = [
            ('all', 'ALL SEGMENTS'),
            ('switch', 'SWITCH SEGMENTS ONLY'),
            ('no_switch', 'NON-SWITCH SEGMENTS ONLY')
        ]

        for category_key, category_name in categories:
            print("\n" + "=" * 200)
            print(f"{category_name}")
            print("=" * 200)

            if category_key in ['all', 'switch']:
                # Print switch detection metrics
                header = (
                    f"{'Model':<30} "
                    f"{'Prec':>8} {'Rec':>8} {'F-β(2)':>8} "
                    f"{'Exact':>8} {'TrueSwt':>10} {'PredSwt':>10} {'ModeAcc':>10}"
                )
                print(header)
                print("-" * len(header))

                for model_name, metrics in results[category_key].items():
                    prec = metrics.get('proximity_precision', 0.0)
                    rec = metrics.get('proximity_recall', 0.0)
                    fbeta = metrics.get('proximity_fbeta2', 0.0)
                    exact = metrics.get('exact_matches', 0)
                    true_sw = metrics.get('total_true_switches', 0)
                    pred_sw = metrics.get('total_pred_switches', 0)
                    mode_acc = metrics.get('mode_accuracy', 0.0)

                    # Add indicator for 3-class models
                    model_display = model_name
                    if self.config.is_three_class_model(model_name):
                        model_display += " [3cls]"
                    else:
                        model_display += " [4cls]"

                    print(f"{model_display:<30} "
                          f"{prec:>8.3f} {rec:>8.3f} {fbeta:>8.3f} "
                          f"{exact:>8d} {true_sw:>10d} {pred_sw:>10d} {mode_acc:>10.3f}")

            else:  # no_switch
                # Print mode accuracy
                header = f"{'Model':<30} {'ModeAcc':>10} {'OverallAcc':>12} {'Segments':>10}"
                print(header)
                print("-" * len(header))

                for model_name, metrics in results[category_key].items():
                    mode_acc = metrics.get('mode_accuracy', 0.0)
                    overall_acc = metrics.get('overall_accuracy', 0.0)
                    seg_count = metrics.get('segment_count', 0)

                    # Add indicator for 3-class models
                    model_display = model_name
                    if self.config.is_three_class_model(model_name):
                        model_display += " [3cls]"
                    else:
                        model_display += " [4cls]"

                    print(f"{model_display:<30} {mode_acc:>10.3f} {overall_acc:>12.3f} {seg_count:>10d}")

    def _print_switch_diagnostics(self, switch_diag: Dict[str, Dict]):
        print("\n" + "=" * 160)
        print("SWITCH SPACING & REPETITION DIAGNOSTICS (predictions; ALL segments)")
        print("=" * 160)

        models = [m for m in self.config.model_names if m in switch_diag]
        col_w = 12
        header = (
            f"{'Model':<32}"
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

            # Add class indicator
            model_display = m
            if self.config.is_three_class_model(m):
                model_display += " [3cls]"
            else:
                model_display += " [4cls]"

            print(
                f"{model_display:<32}"
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

        print(f"\n{'Rank':<6} {'Model':<30} {'All F-β(2)':<12} {'Switch F-β(2)':<15} {'NoSwitch Mode Acc':<20}")
        print("-" * 85)

        for rank, (name, metrics) in enumerate(ranked, 1):
            all_fbeta = metrics['proximity_fbeta2']
            switch_fbeta = results['switch'][name]['proximity_fbeta2'] if name in results['switch'] else 0
            no_switch_acc = results['no_switch'][name]['mode_accuracy'] if name in results['no_switch'] else 0

            # Add class indicator
            model_display = name
            if self.config.is_three_class_model(name):
                model_display += " [3cls]"
            else:
                model_display += " [4cls]"

            print(f"{rank:<6} {model_display:<30} {all_fbeta:<12.3f} {switch_fbeta:<15.3f} {no_switch_acc:<20.3f}")

        # Best in each category
        print("\n" + "=" * 200)
        print("CATEGORY WINNERS")
        print("=" * 200)

        # Best overall
        best_overall = ranked[0]
        best_name = best_overall[0]
        if self.config.is_three_class_model(best_name):
            best_name += " [3cls]"
        else:
            best_name += " [4cls]"
        print(f"🏆 Best Overall: {best_name} (F-β(2) = {best_overall[1]['proximity_fbeta2']:.3f})")

        # Best on switch segments
        switch_ranked = sorted(results['switch'].items(),
                               key=lambda x: x[1]['proximity_fbeta2'],
                               reverse=True)
        if switch_ranked:
            best_switch_name = switch_ranked[0][0]
            if self.config.is_three_class_model(best_switch_name):
                best_switch_name += " [3cls]"
            else:
                best_switch_name += " [4cls]"
            print(
                f"🎯 Best on Switch Segments: {best_switch_name} (F-β(2) = {switch_ranked[0][1]['proximity_fbeta2']:.3f})")

        # Best on non-switch segments
        no_switch_ranked = sorted(results['no_switch'].items(),
                                  key=lambda x: x[1]['mode_accuracy'],
                                  reverse=True)
        if no_switch_ranked:
            best_noswitch_name = no_switch_ranked[0][0]
            if self.config.is_three_class_model(best_noswitch_name):
                best_noswitch_name += " [3cls]"
            else:
                best_noswitch_name += " [4cls]"
            print(
                f"📊 Best on Non-Switch Segments: {best_noswitch_name} (Mode Acc = {no_switch_ranked[0][1]['mode_accuracy']:.3f})")


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

    # Count segments with/without switches (based on 4-class labels)
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
    """Return (positions, types) for labels that are switches {2,3} for 4-class or {1,2} for 3-class."""
    pos, typ = [], []
    # This function is used internally and should be called with the appropriate label set
    # For 4-class: switches are 2 and 3
    # For 3-class: switches are 1 and 2
    for i, l in enumerate(labels):
        if l in (1, 2, 3):  # Covers both 3-class (1,2) and 4-class (2,3)
            # Need context to determine if it's a switch
            # Will be handled by caller
            pos.append(i)
            typ.append(l)
    return pos, typ


def switch_sequence_diagnostics(pred_labels):
    """
    Diagnostics over predicted switches only.
    Works for both 3-class and 4-class models.
    - gaps: list of tokens-between-switches (j - i - 1)
    - adjacent_pairs: gaps == 0
    - single_gap_pairs: gaps == 1
    - same_type_pairs: consecutive switch types equal
    - max_same_type_run: longest run of same switch type in the switch stream
    """
    # Determine if 3-class or 4-class based on max label value
    max_label = max(pred_labels) if pred_labels else 0

    if max_label <= 2:
        # 3-class: switches are 1 and 2
        switch_labels = {1, 2}
    else:
        # 4-class: switches are 2 and 3
        switch_labels = {2, 3}

    idxs = [i for i, l in enumerate(pred_labels) if l in switch_labels]
    types = [l for l in pred_labels if l in switch_labels]

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

if __name__ == "__main__":
    config = EvaluationConfig()

    # You can override the test file path if needed
    # config.test_file = '/path/to/your/test_full.csv'

    results, segment_metrics = run_comprehensive_evaluation(config)