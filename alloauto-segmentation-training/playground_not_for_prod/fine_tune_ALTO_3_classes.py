"""
COMPLETE 3-Class Code-Switching Detection System for Tibetan Text
Converts existing 4-class CSV data to 3-class on-the-fly
Maintains all original ALTO proximity-aware loss behavior

Classes:
  0: Non-switching (merged old classes 0 and 1)
  1: Switch TO Auto (old class 2)
  2: Switch TO Allo (old class 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# ============================================================================
# LABEL CONVERSION FUNCTION
# ============================================================================

def convert_4class_to_3class(labels_4class):
    """
    Convert 4-class labels to 3-class labels:
    OLD -> NEW
    0 (Non-switch Auto) -> 0 (Non-switch)
    1 (Non-switch Allo) -> 0 (Non-switch)
    2 (Switch→Auto) -> 1 (Switch→Auto)
    3 (Switch→Allo) -> 2 (Switch→Allo)
    -100 (padding) -> -100 (unchanged)
    """
    labels_3class = []
    for label in labels_4class:
        if label == 0 or label == 1:
            labels_3class.append(0)  # Merge both non-switch classes
        elif label == 2:
            labels_3class.append(1)  # Switch to Auto
        elif label == 3:
            labels_3class.append(2)  # Switch to Allo
        else:
            labels_3class.append(label)  # Keep -100 and other special tokens
    return labels_3class


# ============================================================================
# DATASET CLASS (converts 4-class to 3-class on-the-fly)
# ============================================================================

class CodeSwitchingDataset3Class(Dataset):
    """
    Dataset for 3-class token-level code-switching.
    Reads 4-class CSV files and converts to 3-class automatically.
    """

    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"  Loading {csv_file}: {len(self.data)} segments")
        print(f"  Converting from 4-class to 3-class labels...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens'].split()
        labels_4class = list(map(int, row['labels'].split(',')))

        # CONVERT 4-class to 3-class
        labels_3class = convert_4class_to_3class(labels_4class)

        encoding = self.tokenize_and_align_labels(tokens, labels_3class)

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(encoding['labels'])
        }

    def tokenize_and_align_labels(self, tokens, labels):
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        word_ids = tokenized_inputs.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx] if word_idx < len(labels) else -100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        tokenized_inputs['labels'] = aligned_labels
        return tokenized_inputs


# ============================================================================
# LOSS FUNCTIONS (3-class versions with full proximity-aware behavior)
# ============================================================================

class SwitchFocusedLoss(nn.Module):
    """
    Loss that primarily cares about detecting switches correctly - 3-CLASS VERSION
    Maintains all original ALTO proximity and segmentation awareness
    """

    def __init__(self, switch_recall_weight=5.0, proximity_tolerance=5,
                 segmentation_penalty=3.0, segmentation_reward=0.3):
        super().__init__()
        self.proximity_tolerance = proximity_tolerance
        self.segmentation_penalty = segmentation_penalty
        self.segmentation_reward = segmentation_reward

        # 3-CLASS weights: ignore non-switch, focus on switches
        self.class_weights = torch.tensor([
            0.1,  # Class 0: Non-switch (merged auto+allo)
            5.0,  # Class 1: Switch to Auto
            5.0   # Class 2: Switch to Allo
        ])

    def check_segmentation_alignment(self, tokens, predictions, b, seq_len):
        """
        Check if switches align with Tibetan segmentation marks / or //
        Returns adjustment factors for each position
        """
        adjustment = torch.ones(seq_len, device=predictions.device)

        for t in range(seq_len):
            if predictions[b, t] >= 1:  # This is a predicted switch (class 1 or 2)

                # Check if switch happens 1-2 positions AFTER segmentation mark (BAD)
                for offset in [1, 2]:
                    if t - offset >= 0:
                        adjustment[t] = self.segmentation_penalty

                # Check if switch happens AT or RIGHT BEFORE segmentation mark (GOOD)
                if t + 1 < seq_len:
                    adjustment[t] = self.segmentation_reward

        return adjustment

    def forward(self, logits, labels, tokens=None):
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device
        self.class_weights = self.class_weights.to(device)

        # Base loss with heavy switch weighting
        base_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='none'
        ).view(batch_size, seq_len)

        predictions = torch.argmax(logits, dim=-1)

        # Apply segmentation alignment if tokens provided
        if tokens is not None:
            for b in range(batch_size):
                seg_adjustment = self.check_segmentation_alignment(tokens, predictions, b, seq_len)
                base_loss[b] *= seg_adjustment

        # Proximity logic - both switch classes treated equally
        for b in range(batch_size):
            # Find all switches (classes 1 and 2)
            true_switches = torch.where(labels[b] >= 1)[0]
            pred_switches = torch.where(predictions[b] >= 1)[0]

            # REWARD predictions near true switches
            for true_pos in true_switches:
                window_start = max(0, true_pos - self.proximity_tolerance)
                window_end = min(seq_len, true_pos + self.proximity_tolerance + 1)
                window_preds = predictions[b, window_start:window_end]
                if torch.any(window_preds >= 1):  # Any switch prediction in window
                    base_loss[b, true_pos] *= 0.1

            # Mild penalty for far predictions
            for pred_pos in pred_switches:
                if len(true_switches) > 0:
                    distances = torch.abs(true_switches - pred_pos)
                    min_distance = torch.min(distances).item()
                    if min_distance > self.proximity_tolerance:
                        base_loss[b, pred_pos] *= 2.0

        valid_mask = (labels != -100).float()
        return (base_loss * valid_mask).sum() / valid_mask.sum()


class ProximityAwareLoss3Class(nn.Module):
    """
    Full proximity-aware loss with logical transition constraints - 3-CLASS VERSION
    Maintains ALL original ALTO behavior including:
    - Proximity tolerance rewards
    - Distance-based penalties
    - Invalid transition penalties
    - Type-specific switch matching
    """

    def __init__(self, switch_loss_weight=30.0, proximity_tolerance=5,
                 distance_decay=0.7, false_positive_penalty=10.0,
                 invalid_transition_penalty=100.0):
        super().__init__()
        self.switch_loss_weight = switch_loss_weight
        self.proximity_tolerance = proximity_tolerance
        self.distance_decay = distance_decay
        self.false_positive_penalty = false_positive_penalty
        self.invalid_transition_penalty = invalid_transition_penalty

        # 3-CLASS weights
        self.class_weights = torch.tensor([
            1.0,  # Class 0: Non-switch
            switch_loss_weight,  # Class 1: Switch to auto
            switch_loss_weight   # Class 2: Switch to allo
        ])

    def forward(self, logits, labels):
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device
        self.class_weights = self.class_weights.to(device)

        # Standard cross-entropy loss with class weights
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='none'
        ).view(batch_size, seq_len)

        valid_mask = (labels != -100).float()
        predictions = torch.argmax(logits, dim=-1)

        # Apply proximity-aware adjustments
        proximity_adjusted_loss = ce_loss.clone()

        for b in range(batch_size):
            # Track current mode to detect invalid transitions
            current_true_mode = 0  # Start in Auto (0=auto, 1=allo)
            current_pred_mode = 0

            for t in range(seq_len):
                if labels[b, t] == -100:
                    continue

                true_label = labels[b, t].item()
                pred_label = predictions[b, t].item()

                # PENALIZE INVALID TRANSITIONS
                if pred_label == 1:  # Predicted Switch to Auto
                    if current_pred_mode == 0:  # Already in Auto - INVALID!
                        proximity_adjusted_loss[b, t] *= self.invalid_transition_penalty
                    current_pred_mode = 0
                elif pred_label == 2:  # Predicted Switch to Allo
                    if current_pred_mode == 1:  # Already in Allo - INVALID!
                        proximity_adjusted_loss[b, t] *= self.invalid_transition_penalty
                    current_pred_mode = 1
                # else: pred_label == 0 (non-switch), mode stays the same

                # REWARD CORRECT LOGICAL TRANSITIONS
                if true_label == 1:  # True Switch to Auto
                    if current_true_mode == 1 and pred_label == 1:  # Correct logic
                        proximity_adjusted_loss[b, t] *= 0.1
                    current_true_mode = 0
                elif true_label == 2:  # True Switch to Allo
                    if current_true_mode == 0 and pred_label == 2:  # Correct logic
                        proximity_adjusted_loss[b, t] *= 0.1
                    current_true_mode = 1
                # else: true_label == 0 (non-switch), mode stays the same

            # TYPE-SPECIFIC proximity logic for switch classes
            true_switches_to_auto = torch.where(labels[b] == 1)[0]
            true_switches_to_allo = torch.where(labels[b] == 2)[0]
            pred_switches_to_auto = torch.where(predictions[b] == 1)[0]
            pred_switches_to_allo = torch.where(predictions[b] == 2)[0]

            # Apply proximity rewards/penalties for Switch→Auto
            for pred_pos in pred_switches_to_auto:
                if len(true_switches_to_auto) > 0:
                    distances = torch.abs(true_switches_to_auto - pred_pos)
                    min_distance = torch.min(distances).item()

                    if min_distance == 0:
                        proximity_adjusted_loss[b, pred_pos] *= 0.1  # Perfect match
                    elif min_distance <= self.proximity_tolerance:
                        # Linear scaling: closer = better reward
                        reward_factor = 0.1 + (min_distance / self.proximity_tolerance) * 0.9
                        proximity_adjusted_loss[b, pred_pos] *= reward_factor
                    else:
                        proximity_adjusted_loss[b, pred_pos] *= self.false_positive_penalty
                else:
                    proximity_adjusted_loss[b, pred_pos] *= self.false_positive_penalty * 2

            # Apply proximity rewards/penalties for Switch→Allo
            for pred_pos in pred_switches_to_allo:
                if len(true_switches_to_allo) > 0:
                    distances = torch.abs(true_switches_to_allo - pred_pos)
                    min_distance = torch.min(distances).item()

                    if min_distance == 0:
                        proximity_adjusted_loss[b, pred_pos] *= 0.1
                    elif min_distance <= self.proximity_tolerance:
                        reward_factor = 0.1 + (min_distance / self.proximity_tolerance) * 0.9
                        proximity_adjusted_loss[b, pred_pos] *= reward_factor
                    else:
                        proximity_adjusted_loss[b, pred_pos] *= self.false_positive_penalty
                else:
                    proximity_adjusted_loss[b, pred_pos] *= self.false_positive_penalty * 2

            # Penalize missed true switches with distance-aware scaling
            for true_pos in true_switches_to_auto:
                if len(pred_switches_to_auto) == 0:
                    proximity_adjusted_loss[b, true_pos] *= 5.0
                else:
                    distances = torch.abs(pred_switches_to_auto - true_pos)
                    min_distance = torch.min(distances).item()
                    if min_distance > self.proximity_tolerance:
                        distance_penalty = 2.0 + (min_distance - self.proximity_tolerance) * 0.3
                        distance_penalty = min(distance_penalty, 8.0)
                        proximity_adjusted_loss[b, true_pos] *= distance_penalty

            for true_pos in true_switches_to_allo:
                if len(pred_switches_to_allo) == 0:
                    proximity_adjusted_loss[b, true_pos] *= 5.0
                else:
                    distances = torch.abs(pred_switches_to_allo - true_pos)
                    min_distance = torch.min(distances).item()
                    if min_distance > self.proximity_tolerance:
                        distance_penalty = 2.0 + (min_distance - self.proximity_tolerance) * 0.3
                        distance_penalty = min(distance_penalty, 8.0)
                        proximity_adjusted_loss[b, true_pos] *= distance_penalty

        # Apply valid mask and compute mean
        total_loss = proximity_adjusted_loss * valid_mask
        loss = total_loss.sum() / valid_mask.sum().clamp(min=1)

        return loss


# ============================================================================
# EVALUATION FUNCTIONS (3-class with full proximity awareness)
# ============================================================================

def evaluate_switch_detection_with_proximity(true_labels, pred_labels, tolerance=5):
    """
    Evaluate switch detection with proximity tolerance and TYPE matching - 3-CLASS VERSION
    Maintains all original ALTO evaluation behavior
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Find switch positions BY TYPE (3-class: 1=auto, 2=allo)
    true_switches_to_auto = np.where(true_labels == 1)[0]
    true_switches_to_allo = np.where(true_labels == 2)[0]
    pred_switches_to_auto = np.where(pred_labels == 1)[0]
    pred_switches_to_allo = np.where(pred_labels == 2)[0]

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


def compute_metrics_for_trainer(eval_pred, tolerance=5):
    """Compute metrics for the trainer - 3-CLASS VERSION"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Flatten all sequences
    all_predictions = predictions.flatten()
    all_labels = labels.flatten()

    # Remove padding
    mask = all_labels != -100
    all_predictions = all_predictions[mask]
    all_labels = all_labels[mask]

    # Basic accuracy
    accuracy = (all_predictions == all_labels).mean()

    # Proximity-aware switch metrics
    switch_metrics = evaluate_switch_detection_with_proximity(
        all_labels, all_predictions, tolerance=tolerance
    )

    return {
        'accuracy': float(accuracy),
        'switch_precision': float(switch_metrics['precision']),
        'switch_recall': float(switch_metrics['recall']),
        'switch_f1': float(switch_metrics['f1']),
        'true_switches': switch_metrics['true_switches'],
        'pred_switches': switch_metrics['pred_switches'],
        'exact_matches': switch_metrics['exact_matches'],
        'proximity_matches': switch_metrics['proximity_matches'],
        'to_auto_precision': float(switch_metrics['to_auto_precision']),
        'to_auto_recall': float(switch_metrics['to_auto_recall']),
        'to_allo_precision': float(switch_metrics['to_allo_precision']),
        'to_allo_recall': float(switch_metrics['to_allo_recall']),
        'true_to_auto': switch_metrics['true_to_auto'],
        'true_to_allo': switch_metrics['true_to_allo'],
        'matched_to_auto': switch_metrics['matched_to_auto'],
        'matched_to_allo': switch_metrics['matched_to_allo']
    }


def apply_transition_constraints(predictions, logits=None):
    """
    Apply logical constraints to predictions - 3-CLASS VERSION
    - If in Auto mode, can only switch to Allo (2)
    - If in Allo mode, can only switch to Auto (1)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    corrected_predictions = predictions.copy()
    current_mode = 0  # Start in Auto by default (0=auto, 1=allo)

    for i in range(len(predictions)):
        pred = predictions[i]

        if pred == -100:  # Skip padding
            continue

        # Check for invalid transitions
        if pred == 1:  # Switch to Auto
            if current_mode == 0:  # Already in Auto - INVALID
                corrected_predictions[i] = 0  # Change to non-switch
            else:  # Was in Allo - VALID
                current_mode = 0

        elif pred == 2:  # Switch to Allo
            if current_mode == 1:  # Already in Allo - INVALID
                corrected_predictions[i] = 0  # Change to non-switch
            else:  # Was in Auto - VALID
                current_mode = 1

        # pred == 0 (non-switch) doesn't change mode

    return corrected_predictions


def print_test_examples_with_constraints(model, tokenizer, test_csv='test_segments.csv',
                                         num_examples=5, tolerance=5):
    """
    Print test examples showing effect of logical constraints - 3-CLASS VERSION
    """
    print("\n" + "=" * 80)
    print(f"SWITCH DETECTION WITH LOGICAL CONSTRAINTS (3-CLASS)")
    print("Rules: Can only switch FROM current mode TO different mode")
    print("=" * 80)

    test_df = pd.read_csv(test_csv)
    device = next(model.parameters()).device
    model.eval()

    # 3-CLASS label names
    label_names = {
        0: 'NonSwitch',
        1: 'SWITCH→Auto',
        2: 'SWITCH→Allo'
    }

    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    for i, idx in enumerate(sample_indices):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()
        true_labels_4class = list(map(int, row['labels'].split(',')))
        # Convert to 3-class for evaluation
        true_labels = convert_4class_to_3class(true_labels_4class)

        print(f"\n--- Example {i + 1} (from {row['source_file'][:40]}...) ---")
        print(f"Segment length: {row['num_tokens']} tokens")

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
            raw_predictions = torch.argmax(outputs.logits, dim=2)

        # Align predictions
        word_ids = tokenizer_output.word_ids()
        aligned_raw = []
        previous_word_idx = None

        for j, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                pred = raw_predictions[0][j].item()
                aligned_raw.append(pred)
            previous_word_idx = word_idx

        # Apply constraints
        final_len = min(len(aligned_raw), len(true_labels), len(tokens))
        aligned_raw = aligned_raw[:final_len]
        aligned_constrained = apply_transition_constraints(aligned_raw)
        true_labels = true_labels[:final_len]
        tokens = tokens[:final_len]

        # Find constraint violations that were corrected
        violations_corrected = []
        current_mode = 0

        for j in range(len(aligned_raw)):
            raw = aligned_raw[j]
            constrained = aligned_constrained[j]

            if raw != constrained:
                violations_corrected.append((j, raw, constrained, current_mode))

            # Update mode based on constrained prediction
            if constrained == 1:
                current_mode = 0  # Switched to Auto
            elif constrained == 2:
                current_mode = 1  # Switched to Allo

        print(f"\nConstraint violations corrected: {len(violations_corrected)}")
        if violations_corrected:
            for pos, raw, fixed, mode in violations_corrected[:5]:
                mode_str = "Auto" if mode == 0 else "Allo"
                print(f"  Pos {pos}: {label_names[raw]} → {label_names[fixed]} (was in {mode_str} mode)")

        # Evaluate both versions
        raw_eval = evaluate_switch_detection_with_proximity(true_labels, aligned_raw, tolerance)
        const_eval = evaluate_switch_detection_with_proximity(true_labels, aligned_constrained, tolerance)

        print(f"\nPerformance comparison:")
        print(f"  {'Metric':<20} {'Raw':<15} {'Constrained':<15}")
        print(f"  {'-' * 50}")
        print(f"  {'Precision':<20} {raw_eval['precision']:<15.3f} {const_eval['precision']:<15.3f}")
        print(f"  {'Recall':<20} {raw_eval['recall']:<15.3f} {const_eval['recall']:<15.3f}")
        print(f"  {'F1':<20} {raw_eval['f1']:<15.3f} {const_eval['f1']:<15.3f}")

    return sample_indices


# ============================================================================
# CUSTOM TRAINER (3-class with full ALTO behavior)
# ============================================================================

class SimpleSwitchTrainer3Class(Trainer):
    """
    Trainer focused on switch detection - 3-CLASS VERSION
    Maintains all original ALTO training behavior
    """

    def __init__(self, *args, **kwargs):
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.get('tokenizer')
        super().__init__(*args, **kwargs)

        # Use the 3-class loss
        self.loss_fn = SwitchFocusedLoss(
            switch_recall_weight=10.0,
            proximity_tolerance=5,
            segmentation_penalty=3.0,
            segmentation_reward=0.3
        )
        self.tokenizer = kwargs.get('tokenizer') or kwargs.get('processing_class')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_ids = inputs.get("input_ids").clone()
        outputs = model(**inputs)
        logits = outputs.get('logits')

        tokens_with_seg_info = self.analyze_segmentation_marks(input_ids)
        loss = self.loss_fn(logits, labels, tokens_with_seg_info)
        return (loss, outputs) if return_outputs else loss

    def analyze_segmentation_marks(self, input_ids):
        """Analyze tokens for Tibetan segmentation marks"""
        batch_size, seq_len = input_ids.shape
        seg_marks = torch.zeros_like(input_ids, dtype=torch.float)

        for b in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b])
            for t, token in enumerate(tokens):
                if token and ('/' in token or '།' in token):
                    seg_marks[b, t] = 1.0

        return seg_marks

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Apply transition constraints during evaluation"""
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        if not prediction_loss_only:
            loss, logits, labels = outputs
            predictions = torch.argmax(logits, dim=-1)

            # Apply transition constraints to each sequence
            batch_size = predictions.shape[0]
            for b in range(batch_size):
                predictions[b] = torch.tensor(
                    apply_transition_constraints(predictions[b]),
                    device=predictions.device
                )

            return (loss, logits, labels)

        return outputs


# ============================================================================
# MAIN TRAINING FUNCTION (uses existing 4-class CSVs)
# ============================================================================

def train_tibetan_code_switching_3class():
    """
    Main training pipeline - 3-CLASS VERSION
    Uses existing 4-class CSV files, converts to 3-class on-the-fly
    Maintains ALL original ALTO proximity-aware behavior
    """
    print("=" * 80)
    print("TIBETAN CODE-SWITCHING DETECTION TRAINING (3-CLASS)")
    print("Converting existing 4-class data to 3-class")
    print("Classes: 0=NonSwitch, 1=Switch→Auto, 2=Switch→Allo")
    print("=" * 80)

    # Use existing CSV files
    train_dataset_file = 'train_segments_clean.csv'  # or 'train_segments.csv'
    val_dataset_file = 'val_segments.csv'
    test_dataset_file = 'test_segments.csv'

    # Check if files exist
    for f in [train_dataset_file, val_dataset_file, test_dataset_file]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found!")
            print(f"Please ensure your 4-class CSV files are in the current directory")
            return

    print(f"\n✓ Using existing datasets:")
    print(f"  Train: {train_dataset_file}")
    print(f"  Val: {val_dataset_file}")
    print(f"  Test: {test_dataset_file}")

    # Initialize model with 3 CLASSES
    print("\n" + "=" * 80)
    print("STEP 1: Initializing 3-class model")
    print("=" * 80)

    model_name = 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
    output_dir = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_3class_converted_10_10'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,  # 3 classes instead of 4
        label2id={'non_switch': 0, 'to_auto': 1, 'to_allo': 2},
        id2label={0: 'non_switch', 1: 'to_auto', 2: 'to_allo'}
    )

    # Initialize with balanced bias for both switch types
    with torch.no_grad():
        model.classifier.bias.data[0] = 0.0   # Non-switch
        model.classifier.bias.data[1] = -1.0  # Switch to auto
        model.classifier.bias.data[2] = -1.0  # Switch to allo

    model = model.to(device)
    print(f"✓ Model initialized with 3 classes")

    # Create datasets (conversion happens automatically)
    print("\n" + "=" * 80)
    print("STEP 2: Loading datasets and converting labels")
    print("=" * 80)

    train_dataset = CodeSwitchingDataset3Class(train_dataset_file, tokenizer)
    val_dataset = CodeSwitchingDataset3Class(val_dataset_file, tokenizer)
    test_dataset = CodeSwitchingDataset3Class(test_dataset_file, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    print(f"✓ Datasets loaded and converted to 3-class")

    # Training setup with SAME hyperparameters as original ALTO
    print("\n" + "=" * 80)
    print("STEP 3: Setting up training (same hyperparameters as 4-class ALTO)")
    print("=" * 80)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=60,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model='switch_f1',
        greater_is_better=True,
        warmup_steps=100,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    # Create trainer
    trainer = SimpleSwitchTrainer3Class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_for_trainer(eval_pred, tolerance=5)
    )

    # Add early stopping
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))
    print(f"✓ Trainer configured with:")
    print(f"  - Proximity tolerance: 5 tokens")
    print(f"  - Switch recall weight: 10.0")
    print(f"  - Segmentation penalty: 3.0")
    print(f"  - Segmentation reward: 0.3")
    print(f"  - Early stopping patience: 5")

    # Train
    print("\n" + "=" * 80)
    print("STEP 4: Training 3-class model with proximity-aware loss")
    print("=" * 80)
    trainer.train()

    # Save final model
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')
    print(f"\n✓ Model saved to: {output_dir}/final_model")

    # Evaluate
    print("\n" + "=" * 80)
    print("STEP 5: Evaluating on test set")
    print("=" * 80)
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\n{'=' * 80}")
    print(f"FINAL 3-CLASS TEST RESULTS")
    print(f"{'=' * 80}")
    print(f"Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Switch F1: {test_results['eval_switch_f1']:.3f}")
    print(f"Switch Precision: {test_results['eval_switch_precision']:.3f}")
    print(f"Switch Recall: {test_results['eval_switch_recall']:.3f}")
    print(f"Exact Matches: {test_results.get('eval_exact_matches', 0)}")
    print(f"Proximity Matches: {test_results.get('eval_proximity_matches', 0)}")
    print(f"True Switches: {test_results['eval_true_switches']}")
    print(f"Predicted Switches: {test_results['eval_pred_switches']}")

    print(f"\nPer-Type Performance:")
    print(f"  Switch→Auto Precision: {test_results.get('eval_to_auto_precision', 0):.3f}")
    print(f"  Switch→Auto Recall: {test_results.get('eval_to_auto_recall', 0):.3f}")
    print(f"  Switch→Allo Precision: {test_results.get('eval_to_allo_precision', 0):.3f}")
    print(f"  Switch→Allo Recall: {test_results.get('eval_to_allo_recall', 0):.3f}")

    # Show examples
    print("\n" + "=" * 80)
    print("STEP 6: Showing test examples with constraints")
    print("=" * 80)
    print_test_examples_with_constraints(model, tokenizer, test_dataset_file, num_examples=3, tolerance=5)

    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Model saved to: {output_dir}/final_model")
    print(f"Final Switch F1: {test_results['eval_switch_f1']:.3f}")

    return trainer, model, tokenizer, test_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING 3-CLASS ALTO TRAINING")
    print("Converting existing 4-class CSV files to 3-class")
    print("=" * 80)
    print("\nLabel conversion mapping:")
    print("  Old Class 0 (Non-switch Auto) → New Class 0 (Non-switch)")
    print("  Old Class 1 (Non-switch Allo) → New Class 0 (Non-switch)")
    print("  Old Class 2 (Switch→Auto) → New Class 1 (Switch→Auto)")
    print("  Old Class 3 (Switch→Allo) → New Class 2 (Switch→Allo)")
    print("\nAll proximity-aware behavior maintained:")
    print("  ✓ 5-token proximity tolerance")
    print("  ✓ Distance-based rewards/penalties")
    print("  ✓ Invalid transition penalties")
    print("  ✓ Type-specific switch matching")
    print("  ✓ Segmentation alignment")

    trainer, model, tokenizer, results = train_tibetan_code_switching_3class()