"""
Compatibility fixes for different versions of transformers library.
This script patches the issues you're experiencing.
"""
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
import os
import sys


#Proximity-aware loss function and metrics for code-switching detection.
#Allows for positional tolerance when detecting switch points.
#FIXED VERSION - Compatible with newer transformers versions.


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from transformers import Trainer

"""
Fixed Proximity-aware loss function and metrics for code-switching detection.
Corrects the UnboundLocalError issue with min_distance variable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


class ProximityAwareLoss(nn.Module):
    """
    Custom loss that gives partial credit for predictions near true switch points.
    """

    def __init__(self, max_distance=10, distance_weights=None, class_weights=None,
                 switch_loss_weight=10.0, proximity_bonus=True, false_positive_penalty=2):
        """
        Args:
            max_distance: Maximum distance (in tokens) to consider for partial credit
            distance_weights: Weights for different distances [exact, 1-off, 2-off, ..., 10-off]
            class_weights: Weights for different classes [no_switch, to_auto, to_allo]
            switch_loss_weight: Extra weight for switch classes (1 and 2)
            proximity_bonus: Whether to give bonus for predictions near true switches
            false_positive_penalty: Penalty multiplier for false positive switch predictions
        """
        super().__init__()
        self.max_distance = max_distance

        # Default distance weights: gradual decay
        if distance_weights is None:
            self.distance_weights = torch.tensor([1.0, 0.99, 0.95, 0.9, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55])
        else:
            self.distance_weights = torch.tensor(distance_weights)

        # Default class weights
        if class_weights is None:
            self.class_weights = torch.tensor([1.0, switch_loss_weight, switch_loss_weight])
        else:
            self.class_weights = torch.tensor(class_weights)

        self.proximity_bonus = proximity_bonus
        self.false_positive_penalty = false_positive_penalty

    def forward(self, logits, labels):
        """
        Compute proximity-aware loss.

        Args:
            logits: Model predictions (batch_size, seq_len, num_classes)
            labels: True labels (batch_size, seq_len)
        """
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device

        # Move weights to device
        self.distance_weights = self.distance_weights.to(device)
        self.class_weights = self.class_weights.to(device)

        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='none'
        ).view(batch_size, seq_len)

        # Mask for valid tokens (not -100)
        valid_mask = (labels != -100).float()

        # Find switch points (classes 1 and 2)
        switch_mask = ((labels == 1) | (labels == 2)).float()
        predictions = torch.argmax(logits, dim=-1)
        pred_switch_mask = ((predictions == 1) | (predictions == 2)).float()

        # Initialize proximity bonus
        proximity_bonus = torch.zeros_like(ce_loss)

        if self.proximity_bonus:
            # For each predicted switch, check if there's a true switch nearby
            for b in range(batch_size):
                # Get positions of true and predicted switches
                true_switch_pos = torch.where(switch_mask[b] == 1)[0]
                pred_switch_pos = torch.where(pred_switch_mask[b] == 1)[0]

                # For each predicted switch
                for pred_pos in pred_switch_pos:
                    if valid_mask[b, pred_pos] == 0:
                        continue

                    # Find nearest true switch
                    if len(true_switch_pos) > 0:
                        distances = torch.abs(true_switch_pos - pred_pos)
                        min_distance = torch.min(distances).item()

                        # If within max_distance, apply bonus (reduce loss)
                        if min_distance <= self.max_distance:
                            weight = self.distance_weights[min(min_distance, len(self.distance_weights) - 1)]
                            # Use weight directly to reduce loss for good predictions
                            proximity_bonus[b, pred_pos] = ce_loss[b, pred_pos] * weight * 0.5

                # For each true switch, penalize if no prediction nearby
                for true_pos in true_switch_pos:
                    if valid_mask[b, true_pos] == 0:
                        continue

                    # Check if there's a predicted switch nearby
                    if len(pred_switch_pos) > 0:
                        distances = torch.abs(pred_switch_pos - true_pos)
                        min_distance = torch.min(distances).item()

                        # If no prediction within max_distance, add extra penalty
                        if min_distance > self.max_distance:
                            ce_loss[b, true_pos] *= 2.0  # Double the loss for missed switches

        # Apply proximity bonus (reduces loss for good predictions)
        ce_loss = ce_loss - proximity_bonus

        # Apply valid mask and compute mean
        masked_loss = ce_loss * valid_mask

        # Add false positive penalty
        false_positive_mask = (pred_switch_mask * (1 - switch_mask)) * valid_mask
        false_positive_penalty = false_positive_mask * self.false_positive_penalty

        # Combine losses and compute scalar mean
        total_loss = masked_loss + false_positive_penalty
        loss = total_loss.sum() / valid_mask.sum().clamp(min=1)

        return loss

class ProximityAwareLoss_old(nn.Module):
    """
    Custom loss that gives partial credit for predictions near true switch points.
    """
    
    def __init__(self, max_distance=10, distance_weights=None, class_weights=None,
                 switch_loss_weight=10.0, proximity_bonus=True, false_positive_penalty=2):
        """
        Args:
            max_distance: Maximum distance (in tokens) to consider for partial credit
            distance_weights: Weights for different distances [exact, 1-off, 2-off, ..., 5-off]
            class_weights: Weights for different classes [no_switch, to_auto, to_allo]
            switch_loss_weight: Extra weight for switch classes (1 and 2)
            proximity_bonus: Whether to give bonus for predictions near true switches
        """
        super().__init__()
        self.max_distance = max_distance
        
        # Default distance weights: exponential decay
        if distance_weights is None:
            # self.distance_weights = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
            # self.distance_weights = torch.tensor([1.0, 0.85, 0.7, 0.55, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1])
            self.distance_weights = torch.tensor([1.0, 0.99, 0.95, 0.9, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55])


        else:
            self.distance_weights = torch.tensor(distance_weights)
            
        # Default class weights
        if class_weights is None:
            self.class_weights = torch.tensor([1.0, switch_loss_weight, switch_loss_weight])
        else:
            self.class_weights = torch.tensor(class_weights)
            
        self.proximity_bonus = proximity_bonus
        self.false_positive_penalty = false_positive_penalty
        
    def forward(self, logits, labels):
        """
        Compute proximity-aware loss.
        
        Args:
            logits: Model predictions (batch_size, seq_len, num_classes)
            labels: True labels (batch_size, seq_len)
        """
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device
        
        # Move weights to device
        self.distance_weights = self.distance_weights.to(device)
        self.class_weights = self.class_weights.to(device)
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes), 
            labels.view(-1), 
            weight=self.class_weights,
            reduction='none'
        ).view(batch_size, seq_len)
        
        # Mask for valid tokens (not -100)
        valid_mask = (labels != -100).float()
        
        # Find switch points (classes 1 and 2)
        switch_mask = ((labels == 1) | (labels == 2)).float()
        predictions = torch.argmax(logits, dim=-1)
        pred_switch_mask = ((predictions == 1) | (predictions == 2)).float()
        
        # Initialize proximity bonus
        proximity_bonus = torch.zeros_like(ce_loss)
        
        if self.proximity_bonus:
            # For each predicted switch, check if there's a true switch nearby
            for b in range(batch_size):
                # Get positions of true and predicted switches
                true_switch_pos = torch.where(switch_mask[b] == 1)[0]
                pred_switch_pos = torch.where(pred_switch_mask[b] == 1)[0]
                
                # For each predicted switch
                for pred_pos in pred_switch_pos:
                    if valid_mask[b, pred_pos] == 0:
                        continue
                        
                    # Find nearest true switch
                    if len(true_switch_pos) > 0:
                        distances = torch.abs(true_switch_pos - pred_pos)
                        min_distance = torch.min(distances).item()
                        
                        # If within max_distance, apply bonus (reduce loss)
                        if min_distance <= self.max_distance:
                            weight = self.distance_weights[min(min_distance, len(self.distance_weights)-1)]
                            # Reduce loss for this prediction based on proximity
                            proximity_bonus[b, pred_pos] = ce_loss[b, pred_pos] * (1 - weight) * 0.5
                
                # For each true switch, penalize if no prediction nearby
                for true_pos in true_switch_pos:
                    if valid_mask[b, true_pos] == 0:
                        continue
                        
                    # Check if there's a predicted switch nearby
                    if len(pred_switch_pos) > 0:
                        distances = torch.abs(pred_switch_pos - true_pos)
                        min_distance = torch.min(distances).item()
                        
                        # If no prediction within max_distance, add extra penalty
                        if min_distance > self.max_distance:
                            ce_loss[b, true_pos] *= 2.0  # Double the loss for missed switches
        
        # Apply proximity bonus
        # ce_loss = ce_loss - proximity_bonus
        if min_distance <= self.max_distance:
            weight = self.distance_weights[min(min_distance, len(self.distance_weights) - 1)]
            # Use weight directly, not (1 - weight)
            proximity_bonus[b, pred_pos] = ce_loss[b, pred_pos] * weight * 0.5

        # Later, apply the bonus (reduces loss for good predictions)
        ce_loss = ce_loss - proximity_bonus
        # Apply valid mask and compute mean
        masked_loss = ce_loss * valid_mask

        # Add false positive penalty
        false_positive_mask = (pred_switch_mask * (1 - switch_mask)) * valid_mask
        false_positive_penalty = false_positive_mask * self.false_positive_penalty

        # Combine losses and compute scalar mean
        total_loss = masked_loss + false_positive_penalty
        loss = total_loss.sum() / valid_mask.sum().clamp(min=1)
        
        return loss


class ProximityAwareTrainer(Trainer):
    """
    Custom trainer that uses proximity-aware loss.
    Compatible with different transformers versions.
    """
    
    def __init__(self, max_distance=10, distance_weights=None,
                 switch_loss_weight=10.0, *args, **kwargs):
        # Handle the tokenizer/processing_class deprecation
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.get('tokenizer')
        
        super().__init__(*args, **kwargs)
        self.proximity_loss = ProximityAwareLoss(
            max_distance=max_distance,
            distance_weights=distance_weights,
            switch_loss_weight=switch_loss_weight
        )
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with compatibility for different transformers versions.
        The num_items_in_batch parameter is used in newer versions but we can ignore it.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Use proximity-aware loss
        loss = self.proximity_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# Add the rest of the functions from the original file
class ProximityAwareMetrics:
    """
    Evaluation metrics that consider proximity for switch detection.
    """
    
    def __init__(self, max_distance=10):
        self.max_distance = max_distance
        
    def compute_proximity_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics with proximity tolerance.
        
        Returns:
            Dictionary with various metrics
        """
        # Flatten and remove invalid labels
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        # Standard metrics
        correct = (predictions == labels).sum()
        total = len(labels)
        accuracy = correct / total if total > 0 else 0
        
        # Switch detection metrics with proximity
        results = {
            'accuracy': accuracy,
            'total_true_switches': 0,
            'total_pred_switches': 0,
            'exact_switch_matches': 0,
            'proximity_matches': {i: 0 for i in range(1, self.max_distance + 1)},
            'missed_switches': 0,
            'false_switches': 0
        }
        
        # Process each sequence
        true_switches = []
        pred_switches = []
        
        # Find all switch positions
        for i, (true_label, pred_label) in enumerate(zip(labels, predictions)):
            if true_label in [1, 2]:
                true_switches.append((i, true_label))
            if pred_label in [1, 2]:
                pred_switches.append((i, pred_label))
        
        results['total_true_switches'] = len(true_switches)
        results['total_pred_switches'] = len(pred_switches)
        
        # Match predictions to true switches
        matched_true = set()
        matched_pred = set()
        
        # First pass: exact matches
        for i, (true_pos, true_class) in enumerate(true_switches):
            for j, (pred_pos, pred_class) in enumerate(pred_switches):
                if true_pos == pred_pos and true_class == pred_class:
                    results['exact_switch_matches'] += 1
                    matched_true.add(i)
                    matched_pred.add(j)
        
        # Second pass: proximity matches
        for i, (true_pos, true_class) in enumerate(true_switches):
            if i in matched_true:
                continue
                
            best_distance = float('inf')
            best_pred_idx = None
            
            for j, (pred_pos, pred_class) in enumerate(pred_switches):
                if j in matched_pred:
                    continue
                    
                distance = abs(true_pos - pred_pos)
                if distance <= self.max_distance and distance < best_distance:
                    # Consider class match too
                    if true_class == pred_class or distance <= 2:  # More forgiving for very close matches
                        best_distance = distance
                        best_pred_idx = j
            
            if best_pred_idx is not None:
                if best_distance > 0 and best_distance <= self.max_distance:
                    results['proximity_matches'][int(best_distance)] += 1
                elif best_distance == 0:
                    # This should have been caught in the exact matches phase
                    results['exact_switch_matches'] += 1
                matched_true.add(i)
                matched_pred.add(best_pred_idx)
        
        # Count missed and false switches
        results['missed_switches'] = len(true_switches) - len(matched_true)
        results['false_switches'] = len(pred_switches) - len(matched_pred)
        
        # Calculate proximity-aware F1 scores
        if results['total_true_switches'] > 0:
            # Recall with proximity
            proximity_recall_scores = []
            for dist in range(self.max_distance + 1):
                if dist == 0:
                    matches = results['exact_switch_matches']
                else:
                    matches = results['proximity_matches'].get(dist, 0)
                # weight = 1.0 - (dist * 0.15)  # Decreasing weight with distance
                weight = 1.0 - (dist * 0.08)
                proximity_recall_scores.append(matches * weight)
            
            results['proximity_recall'] = sum(proximity_recall_scores) / results['total_true_switches']
        else:
            results['proximity_recall'] = 0.0
        
        if results['total_pred_switches'] > 0:
            # Precision with proximity
            total_valid_predictions = (results['exact_switch_matches'] + 
                                     sum(results['proximity_matches'].values()))
            results['proximity_precision'] = total_valid_predictions / results['total_pred_switches']
        else:
            results['proximity_precision'] = 0.0
        
        # F1 score
        if results['proximity_precision'] + results['proximity_recall'] > 0:
            results['proximity_f1'] = (2 * results['proximity_precision'] * results['proximity_recall'] / 
                                      (results['proximity_precision'] + results['proximity_recall']))
        else:
            results['proximity_f1'] = 0.0
        
        return results


def compute_metrics_with_proximity(eval_pred, max_distance=10):
    """
    Compute metrics function for use with HuggingFace Trainer.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Initialize metrics calculator
    metrics_calc = ProximityAwareMetrics(max_distance=max_distance)
    
    # Compute metrics for the batch
    all_metrics = []
    for pred_seq, label_seq in zip(predictions, labels):
        seq_metrics = metrics_calc.compute_proximity_metrics(pred_seq, label_seq)
        all_metrics.append(seq_metrics)
    
    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], dict):
            # Handle nested dictionaries (proximity_matches)
            aggregated[key] = {}
            for subkey in all_metrics[0][key].keys():
                values = [m[key][subkey] for m in all_metrics]
                aggregated[key][subkey] = sum(values) / len(values)
        else:
            values = [m[key] for m in all_metrics]
            aggregated[key] = sum(values) / len(values)
    
    # Flatten for trainer
    flat_metrics = {
        'accuracy': aggregated['accuracy'],
        'proximity_f1': aggregated['proximity_f1'],
        'proximity_recall': aggregated['proximity_recall'],
        'proximity_precision': aggregated['proximity_precision'],
        'exact_matches': aggregated['exact_switch_matches'],
        'missed_switches': aggregated['missed_switches'],
        'false_switches': aggregated['false_switches']
    }
    
    # Add distance breakdown
    for dist in range(1, max_distance + 1):
        flat_metrics[f'matches_at_{dist}_words'] = aggregated['proximity_matches'][dist]
    
    return flat_metrics

    # Save the fixed version
    with open('proximity_aware_loss_fixed.py', 'w') as f:
        f.write(content)

    print("✓ Created proximity_aware_loss_fixed.py with compatibility fixes")


#Working training script with all compatibility fixes applied.


import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from torch.utils.data import Dataset


from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from torch.utils.data import Dataset, DataLoader
import json
from collections import defaultdict


def prepare_for_bert_training_with_test(sequences_csv, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare data for BERT fine-tuning with train/validation/test split.
    Maintains original order - no shuffling.

    Args:
        sequences_csv: Path to sequences CSV file
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for test (default 0.15)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"

    df = pd.read_csv(sequences_csv)

    # Keep original order - no shuffling
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Sequential split
    train_data = df[:n_train]
    val_data = df[n_train:n_train + n_val]
    test_data = df[n_train + n_val:]

    # Save splits
    train_data.to_csv('train_sequences.csv', index=False)
    val_data.to_csv('val_sequences.csv', index=False)
    test_data.to_csv('test_sequences.csv', index=False)

    print(f"\n=== Sequential Train/Val/Test Split ===")
    print(f"Total sequences: {n_total}")
    print(f"Training sequences: {len(train_data)} (indices 0-{n_train - 1})")
    print(f"Validation sequences: {len(val_data)} (indices {n_train}-{n_train + n_val - 1})")
    print(f"Test sequences: {len(test_data)} (indices {n_train + n_val}-{n_total - 1})")

    # Show distribution of switches in each split
    print(f"\nSequences with switches:")
    print(
        f"  Train: {train_data['contains_switch'].sum()} / {len(train_data)} ({train_data['contains_switch'].sum() / len(train_data) * 100:.1f}%)")
    print(
        f"  Val: {val_data['contains_switch'].sum()} / {len(val_data)} ({val_data['contains_switch'].sum() / len(val_data) * 100:.1f}%)")
    print(
        f"  Test: {test_data['contains_switch'].sum()} / {len(test_data)} ({test_data['contains_switch'].sum() / len(test_data) * 100:.1f}%)")

    # Calculate label distribution for each split
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        label_counts = defaultdict(int)
        total_tokens = 0

        for _, row in split_data.iterrows():
            labels = list(map(int, row['labels'].split(',')))
            for label in labels:
                label_counts[label] += 1
                total_tokens += 1

        print(f"\n{split_name} label distribution:")
        for i in range(3):
            pct = (label_counts[i] / total_tokens * 100) if total_tokens > 0 else 0
            print(f"  Class {i}: {label_counts[i]} ({pct:.2f}%)")

    # Warning if distribution is very unbalanced
    train_switch_ratio = train_data['contains_switch'].sum() / len(train_data)
    test_switch_ratio = test_data['contains_switch'].sum() / len(test_data)

    if abs(train_switch_ratio - test_switch_ratio) > 0.2:
        print("\n⚠️  WARNING: Switch distribution differs significantly between train and test!")
        print("Consider using stratified sequential splitting if needed.")

    return train_data, val_data, test_data

def evaluate_on_test_set(model, tokenizer, test_csv='test_sequences.csv', max_distance=10):
    """
    Comprehensive evaluation on test set with proximity-aware metrics.
    """
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    # Load test data
    test_df = pd.read_csv(test_csv)
    device = next(model.parameters()).device
    model.eval()

    # Initialize metrics
    all_predictions = []
    all_labels = []
    detailed_results = []

    # Process each test sequence
    for idx, row in test_df.iterrows():
        tokens = row['tokens'].split()
        true_labels = list(map(int, row['labels'].split(',')))

        # Tokenize
        tokenizer_output = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Get word alignment
        word_ids = tokenizer_output.word_ids()
        inputs = {k: v.to(device) for k, v in tokenizer_output.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            probs = torch.softmax(outputs.logits, dim=2)

        # Align predictions with original tokens
        aligned_predictions = []
        aligned_probs = []
        previous_word_idx = None

        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                pred = predictions[0][i].item()
                aligned_predictions.append(pred)
                aligned_probs.append(probs[0][i].cpu().numpy())
            previous_word_idx = word_idx

        # Trim to match token length
        aligned_predictions = aligned_predictions[:len(tokens)]
        final_len = min(len(aligned_predictions), len(true_labels), len(tokens))
        if final_len > 0:
            all_predictions.extend(aligned_predictions[:final_len])
            all_labels.extend(true_labels[:final_len])
        else:
            print(f"Warning: Skipping sequence {idx} - no valid alignments")
        aligned_labels = true_labels[:len(tokens)]

        all_predictions.extend(aligned_predictions)
        all_labels.extend(aligned_labels)

        # Store detailed results for this sequence
        sequence_result = {
            'sequence_idx': idx,
            'tokens': tokens,
            'true_labels': aligned_labels,
            'predictions': aligned_predictions,
            'has_true_switches': any(l > 0 for l in aligned_labels),
            'has_pred_switches': any(p > 0 for p in aligned_predictions)
        }

        # Find switches
        true_switches = [(i, l) for i, l in enumerate(aligned_labels) if l > 0]
        pred_switches = [(i, p) for i, p in enumerate(aligned_predictions) if p > 0]

        sequence_result['true_switches'] = true_switches
        sequence_result['pred_switches'] = pred_switches

        detailed_results.append(sequence_result)

    # Compute proximity-aware metrics
    metrics_calculator = ProximityAwareMetrics(max_distance=max_distance)

    # Convert to arrays and ensure same length
    all_predictions_array = np.array(all_predictions)
    all_labels_array = np.array(all_labels)

    print(f"DEBUG: Before trimming - Predictions: {len(all_predictions_array)}, Labels: {len(all_labels_array)}")

    # CRITICAL FIX: Ensure same dimensions
    min_len = min(len(all_predictions_array), len(all_labels_array))
    all_predictions_array = all_predictions_array[:min_len]
    all_labels_array = all_labels_array[:min_len]

    print(f"DEBUG: After trimming - Both arrays: {min_len} tokens")

    # Now compute metrics
    metrics_calculator = ProximityAwareMetrics(max_distance=max_distance)
    proximity_metrics = metrics_calculator.compute_proximity_metrics(
        all_predictions_array,
        all_labels_array
    )

    # Print results
    print("\n=== Overall Test Metrics ===")
    print(f"Total test tokens: {len(all_labels)}")
    print(f"Accuracy: {proximity_metrics['accuracy']:.3f}")
    print(f"\nProximity-aware metrics:")
    print(f"  F1 Score: {proximity_metrics['proximity_f1']:.3f}")
    print(f"  Precision: {proximity_metrics['proximity_precision']:.3f}")
    print(f"  Recall: {proximity_metrics['proximity_recall']:.3f}")

    print(f"\n=== Switch Detection Performance ===")
    print(f"Total true switches: {proximity_metrics['total_true_switches']}")
    print(f"Total predicted switches: {proximity_metrics['total_pred_switches']}")
    print(f"Exact matches: {proximity_metrics['exact_switch_matches']}")

    print("\nDistance breakdown:")
    for dist in range(1, max_distance + 1):
        matches = proximity_metrics['proximity_matches'][dist]
        print(f"  {dist} word{'s' if dist > 1 else ''} off: {matches}")

    print(f"\nMissed switches: {proximity_metrics['missed_switches']}")
    print(f"False switches: {proximity_metrics['false_switches']}")

    # Per-class performance
    # Per-class performance
    print("\n=== Per-Class Performance ===")
    from sklearn.metrics import classification_report

    # Convert to numpy arrays and ensure same length
    all_predictions_np = np.array(all_predictions)
    all_labels_np = np.array(all_labels)

    # Ensure same length
    min_len = min(len(all_predictions_np), len(all_labels_np))
    all_predictions_np = all_predictions_np[:min_len]
    all_labels_np = all_labels_np[:min_len]

    # Filter out -100 labels if present
    if -100 in all_labels_np:
        valid_mask = all_labels_np != -100
        filtered_labels = all_labels_np[valid_mask]
        filtered_predictions = all_predictions_np[valid_mask]
    else:
        # No -100 labels, use as is
        filtered_labels = all_labels_np
        filtered_predictions = all_predictions_np

    if len(filtered_labels) > 0:
        print(classification_report(
            filtered_labels,
            filtered_predictions,
            labels=[0, 1, 2],
            target_names=['No Switch', 'To Auto', 'To Allo'],
            digits=3
        ))
    else:
        print("No valid labels found for classification report")


    print(classification_report(
        filtered_labels,
        filtered_predictions,
        labels=[0, 1, 2],
        target_names=['No Switch', 'To Auto', 'To Allo'],
        digits=3
    ))

    # Example predictions
    print("\n=== Example Predictions ===")
    # Show a few examples with switches
    examples_shown = 0
    for result in detailed_results:
        if result['has_true_switches'] and examples_shown < 3:
            print(f"\nExample {examples_shown + 1}:")
            print(f"Tokens: {' '.join(result['tokens'][:50])}...")  # First 50 tokens
            print(f"True switches: {result['true_switches'][:5]}")  # First 5 switches
            print(f"Predicted switches: {result['pred_switches'][:5]}")
            examples_shown += 1

    # Save detailed results
    results_summary = {
        'overall_metrics': proximity_metrics,
        'total_sequences': len(test_df),
        'sequences_with_true_switches': sum(1 for r in detailed_results if r['has_true_switches']),
        'sequences_with_pred_switches': sum(1 for r in detailed_results if r['has_pred_switches']),
        'detailed_results': detailed_results[:10]  # Save first 10 for inspection
    }

    with open('test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nDetailed results saved to test_results.json")

    return proximity_metrics, detailed_results


# Include your CodeSwitchingDataset class here
class CodeSwitchingDataset(Dataset):
    """Dataset for token-level code-switching."""

    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens'].split()
        labels = list(map(int, row['labels'].split(',')))

        # Tokenize and align labels
        encoding = self.tokenize_and_align_labels(tokens, labels)

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


"""
Fixed training and evaluation pipeline for code-switching detection.
Addresses data imbalance, evaluation bugs, and training issues.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json


def stratified_split_for_sequences(sequences_csv, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create stratified train/val/test split ensuring switch examples in all splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"

    df = pd.read_csv(sequences_csv)

    # Separate sequences with and without switches
    switch_sequences = df[df['contains_switch'] == 1]
    no_switch_sequences = df[df['contains_switch'] == 0]

    print(f"Total sequences: {len(df)}")
    print(f"  With switches: {len(switch_sequences)}")
    print(f"  Without switches: {len(no_switch_sequences)}")

    # Split each group separately to maintain distribution
    def split_group(group_df, train_r, val_r, test_r):
        n = len(group_df)
        n_train = int(n * train_r)
        n_val = int(n * val_r)

        train = group_df.iloc[:n_train]
        val = group_df.iloc[n_train:n_train + n_val]
        test = group_df.iloc[n_train + n_val:]

        return train, val, test

    # Split both groups
    switch_train, switch_val, switch_test = split_group(switch_sequences, train_ratio, val_ratio, test_ratio)
    no_switch_train, no_switch_val, no_switch_test = split_group(no_switch_sequences, train_ratio, val_ratio,
                                                                 test_ratio)

    # Combine and shuffle within each split
    train_data = pd.concat([switch_train, no_switch_train]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_data = pd.concat([switch_val, no_switch_val]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = pd.concat([switch_test, no_switch_test]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save splits
    train_data.to_csv('train_sequences.csv', index=False)
    val_data.to_csv('val_sequences.csv', index=False)
    test_data.to_csv('test_sequences.csv', index=False)

    print(f"\n=== Stratified Train/Val/Test Split ===")
    print(f"Training: {len(train_data)} sequences")
    print(f"  With switches: {train_data['contains_switch'].sum()} ({train_data['contains_switch'].mean() * 100:.1f}%)")
    print(f"Validation: {len(val_data)} sequences")
    print(f"  With switches: {val_data['contains_switch'].sum()} ({val_data['contains_switch'].mean() * 100:.1f}%)")
    print(f"Test: {len(test_data)} sequences")
    print(f"  With switches: {test_data['contains_switch'].sum()} ({test_data['contains_switch'].mean() * 100:.1f}%)")

    # Analyze label distribution
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        label_counts = defaultdict(int)
        total_switches = 0

        for _, row in split_data.iterrows():
            labels = list(map(int, row['labels'].split(',')))
            for label in labels:
                label_counts[label] += 1
                if label > 0:
                    total_switches += 1

        print(f"\n{split_name} switch distribution:")
        print(f"  Total switch labels: {total_switches}")
        print(f"  To Auto (1): {label_counts[1]}")
        print(f"  To Allo (2): {label_counts[2]}")

    return train_data, val_data, test_data


def evaluate_on_test_set_fixed(model, tokenizer, test_csv='test_sequences.csv', max_distance=10):
    """
    Fixed evaluation function without double-counting bug.
    """
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    # Load test data
    test_df = pd.read_csv(test_csv)
    device = next(model.parameters()).device
    model.eval()

    # Initialize metrics
    all_predictions = []
    all_labels = []
    detailed_results = []

    # Process each test sequence
    for idx, row in test_df.iterrows():
        tokens = row['tokens'].split()
        true_labels = list(map(int, row['labels'].split(',')))

        # Tokenize
        tokenizer_output = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Get word alignment
        word_ids = tokenizer_output.word_ids()
        inputs = {k: v.to(device) for k, v in tokenizer_output.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            probs = torch.softmax(outputs.logits, dim=2)

        # Align predictions with original tokens
        aligned_predictions = []
        aligned_probs = []
        previous_word_idx = None

        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                pred = predictions[0][i].item()
                aligned_predictions.append(pred)
                aligned_probs.append(probs[0][i].cpu().numpy())
            previous_word_idx = word_idx

        # Trim to match token length - ONLY ONCE!
        final_len = min(len(aligned_predictions), len(true_labels), len(tokens))
        aligned_predictions = aligned_predictions[:final_len]
        aligned_labels = true_labels[:final_len]

        if final_len > 0:
            all_predictions.extend(aligned_predictions)
            all_labels.extend(aligned_labels)

        # Store detailed results for this sequence
        sequence_result = {
            'sequence_idx': idx,
            'tokens': tokens[:final_len],
            'true_labels': aligned_labels,
            'predictions': aligned_predictions,
            'has_true_switches': any(l > 0 for l in aligned_labels),
            'has_pred_switches': any(p > 0 for p in aligned_predictions)
        }

        # Find switches
        true_switches = [(i, l) for i, l in enumerate(aligned_labels) if l > 0]
        pred_switches = [(i, p) for i, p in enumerate(aligned_predictions) if p > 0]

        sequence_result['true_switches'] = true_switches
        sequence_result['pred_switches'] = pred_switches

        detailed_results.append(sequence_result)

    # Import ProximityAwareMetrics (assuming it's defined elsewhere)
    # from proximity_aware_loss_fixed import ProximityAwareMetrics

    # Convert to arrays
    all_predictions_array = np.array(all_predictions)
    all_labels_array = np.array(all_labels)

    print(f"Total tokens processed: {len(all_predictions_array)}")

    # Compute metrics
    metrics_calculator = ProximityAwareMetrics(max_distance=max_distance)
    proximity_metrics = metrics_calculator.compute_proximity_metrics(
        all_predictions_array,
        all_labels_array
    )

    # Print results
    print("\n=== Overall Test Metrics ===")
    print(f"Total test tokens: {len(all_labels_array)}")
    print(f"Accuracy: {proximity_metrics['accuracy']:.3f}")
    print(f"\nProximity-aware metrics:")
    print(f"  F1 Score: {proximity_metrics['proximity_f1']:.3f}")
    print(f"  Precision: {proximity_metrics['proximity_precision']:.3f}")
    print(f"  Recall: {proximity_metrics['proximity_recall']:.3f}")

    print(f"\n=== Switch Detection Performance ===")
    print(f"Total true switches: {proximity_metrics['total_true_switches']}")
    print(f"Total predicted switches: {proximity_metrics['total_pred_switches']}")
    print(f"Exact matches: {proximity_metrics['exact_switch_matches']}")

    # Show distance breakdown
    print("\nDistance breakdown:")
    for dist in range(1, max_distance + 1):
        matches = proximity_metrics['proximity_matches'].get(dist, 0)
        print(f"  {dist} word{'s' if dist > 1 else ''} off: {matches}")

    print(f"\nMissed switches: {proximity_metrics['missed_switches']}")
    print(f"False switches: {proximity_metrics['false_switches']}")

    # Per-class performance
    if proximity_metrics['total_true_switches'] > 10:  # Only show if enough examples
        from sklearn.metrics import classification_report
        print("\n=== Per-Class Performance ===")
        print(classification_report(
            all_labels_array,
            all_predictions_array,
            labels=[0, 1, 2],
            target_names=['No Switch', 'To Auto', 'To Allo'],
            digits=3,
            zero_division=0
        ))

    # Show examples with switches
    print("\n=== Example Predictions ===")
    examples_shown = 0
    for result in detailed_results:
        if result['has_true_switches'] and examples_shown < 5:
            print(f"\nExample {examples_shown + 1}:")
            print(f"Tokens (first 30): {' '.join(result['tokens'][:30])}...")
            print(f"True switches: {result['true_switches'][:5]}")
            print(f"Predicted switches: {result['pred_switches'][:5]}")
            examples_shown += 1

    # Save results
    results_summary = {
        'overall_metrics': proximity_metrics,
        'total_sequences': len(test_df),
        'sequences_with_true_switches': sum(1 for r in detailed_results if r['has_true_switches']),
        'sequences_with_pred_switches': sum(1 for r in detailed_results if r['has_pred_switches'])
    }

    with open('test_results_fixed.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nDetailed results saved to test_results_fixed.json")

    return proximity_metrics, detailed_results


def augment_training_data(train_csv='train_sequences.csv', augmentation_factor=2):
    """
    Augment training data to balance switch vs no-switch sequences.
    """
    train_df = pd.read_csv(train_csv)

    # Analyze current distribution
    switch_sequences = train_df[train_df['contains_switch'] == 1]
    no_switch_sequences = train_df[train_df['contains_switch'] == 0]

    print(f"\nOriginal training data:")
    print(f"  With switches: {len(switch_sequences)}")
    print(f"  Without switches: {len(no_switch_sequences)}")

    # Duplicate sequences with switches
    augmented_switch_sequences = pd.concat([switch_sequences] * augmentation_factor)

    # Combine with a subset of no-switch sequences for balance
    # Keep ratio around 1:3 (switch:no-switch)
    n_no_switch_to_keep = min(len(no_switch_sequences), len(augmented_switch_sequences) * 3)
    balanced_no_switch = no_switch_sequences.sample(n=n_no_switch_to_keep, random_state=42)

    # Combine and shuffle
    augmented_train = pd.concat([augmented_switch_sequences, balanced_no_switch])
    augmented_train = augmented_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save augmented data
    augmented_train.to_csv('train_sequences_augmented.csv', index=False)

    print(f"\nAugmented training data:")
    print(f"  Total sequences: {len(augmented_train)}")
    print(
        f"  With switches: {augmented_train['contains_switch'].sum()} ({augmented_train['contains_switch'].mean() * 100:.1f}%)")

    return augmented_train


def train_and_evaluate_full_pipeline():
    """Complete pipeline: process data, split, train, and evaluate on test set."""

    from organize_allo_auto_code_switching_3_classes import process_tibetan_3class

    # Step 1: Process your .docx file
    print("Step 1: Processing .docx file...")
    token_df, seq_df = process_tibetan_3class(
        'classify_allo_auto/data/Nicola_Bajetta_rNam_gsum_bshad_pa_Auto_vs_Allo_signals_alo_and_auto_cleaned.docx',
        'classify_allo_auto/data/tokens_3class.csv',
        'classify_allo_auto/data/sequences_3class.csv',
        sequence_length=512
    )

    # Step 2: Create STRATIFIED train/val/test split
    print("\nStep 2: Creating stratified train/val/test split...")
    train_df, val_df, test_df = stratified_split_for_sequences(
        'classify_allo_auto/data/sequences_3class.csv',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Step 3: Augment training data (optional but recommended)
    print("\nStep 3: Augmenting training data...")
    augmented_train = augment_training_data(
        'train_sequences.csv',
        augmentation_factor=3  # Triplicate switch sequences
    )

    # Step 4: Initialize model and tokenizer
    print("\nStep 4: Initializing model and tokenizer...")
    model_name = 'OMRIDRORI/mbert-tibetan-continual-unicode-240k'
    # model_name = 'bert-base-multilingual-cased'
    output_dir = './classify_allo_auto/proximity_cs_model_with_test'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tokenizer and model FIRST
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,
        label2id={'no_switch': 0, 'to_auto': 1, 'to_allo': 2},
        id2label={0: 'no_switch', 1: 'to_auto', 2: 'to_allo'}
    )
    model = model.to(device)

    # Step 5: Create datasets AFTER tokenizer is initialized
    print("\nStep 5: Creating datasets...")
    train_dataset = CodeSwitchingDataset('train_sequences_augmented.csv', tokenizer)
    val_dataset = CodeSwitchingDataset('val_sequences.csv', tokenizer)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Step 6: Set up training
    print("\nStep 6: Setting up training...")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=25,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model='proximity_f1',
        greater_is_better=True,
        warmup_steps=500,
        save_total_limit=1,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
    )

    # Custom metrics function
    def compute_metrics(eval_pred):
        return compute_metrics_with_proximity(eval_pred, max_distance=10)

    # Initialize trainer with higher switch_loss_weight for imbalanced data
    trainer = ProximityAwareTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        max_distance=10,
        distance_weights=[1.0, 0.99, 0.95, 0.9, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55],
        switch_loss_weight=15  # Increased from 5 to help with imbalanced data
    )

    # Step 7: Train
    print("\nStep 7: Starting training with proximity-aware loss...")
    trainer.train()

    # Save the model
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')

    # Step 8: Evaluate on test set with fixed evaluation
    print("\nStep 8: Evaluating on test set...")
    test_metrics, test_results = evaluate_on_test_set_fixed(
        model,
        tokenizer,
        'test_sequences.csv',
        max_distance=10
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {output_dir}/final_model")
    print(f"Test F1 Score: {test_metrics['proximity_f1']:.3f}")

    # Additional diagnostic information
    if test_metrics['proximity_f1'] < 0.1:
        print("\n⚠️ WARNING: Very low F1 score detected!")
        print("Possible causes:")
        print("1. Severe class imbalance - consider more aggressive augmentation")
        print("2. Model not learning to detect switches - try increasing switch_loss_weight to 20-30")
        print("3. Data quality issues - verify your labeled data")
        print("\nRecommendations:")
        print("- Check if model is predicting any switches at all during training")
        print("- Monitor training loss to ensure it's decreasing")
        print("- Consider using focal loss instead of weighted cross-entropy")

    import ipdb
    ipdb.set_trace()

    return trainer, model, tokenizer, test_metrics

if __name__ == "__main__":
    # train_and_evaluate_full_pipeline_fixed()
    train_and_evaluate_full_pipeline()