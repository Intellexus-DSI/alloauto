"""
ALTO Architecture WITHOUT Segmentation Mark Awareness
======================================================
This is your EXACT ALTO code with ONLY the segmentation mark
reward/punishment removed. Everything else is identical:
- Same proximity-aware loss
- Same logical constraints
- Same class weights
- Same training parameters
Just NO "/" or "།" detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import re
import os
import ssl
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change as needed
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================================
# MODELS TO TRAIN
# ============================================================================

MODELS_CONFIG = {
    # 'hfl/cino-base-v2': 'cino_base_v2_no_seg',
    # 'bert-base-multilingual-cased': 'mbert_cased_no_seg',
    # 'xlm-roberta-base': 'xlm_roberta_no_seg',
    # 'sangjeedondrub/tibetan-roberta-base': 'tibetan_roberta_no_seg',
    'OMRIDRORI/mbert-tibetan-continual-wylie-final': 'mbert_tibetan_wylie_no_seg_23_10',
}

# Directories
DATA_DIR = 'dataset/preprocessed_augmented'
# for ALTO keep this
OUTPUT_BASE_DIR = './alloauto-segmentation-training/fine_tuned_ALTO_models'
# for benchmark:
# OUTPUT_BASE_DIR = './alloauto-segmentation-training/benchmark_models_ALTO_no_segmentation'


# ============================================================================
# YOUR EXACT DATASET CLASS (unchanged)
# ============================================================================

class CodeSwitchingDataset4Class(Dataset):
    """Your exact dataset class for 4-class code-switching."""

    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loaded {len(self.data)} segments from {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get tokens and labels
        tokens = row['tokens'].split()
        labels = [int(l) for l in row['labels'].split(',')]

        # Ensure alignment
        if len(tokens) != len(labels):
            min_len = min(len(tokens), len(labels))
            tokens = tokens[:min_len]
            labels = labels[:min_len]

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append(-100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        encoding["labels"] = torch.tensor(aligned_labels)

        return {key: val.squeeze() for key, val in encoding.items()}


# ============================================================================
# MODIFIED LOSS FUNCTION - NO SEGMENTATION MARKS
# ============================================================================

class SwitchFocusedLossNoSegmentation(nn.Module):
    """
    Your EXACT SwitchFocusedLoss but WITHOUT segmentation mark detection.
    Removed: segmentation_penalty, segmentation_reward, check_segmentation_alignment
    """

    def __init__(self, switch_recall_weight=10.0, proximity_tolerance=5):
        super().__init__()
        self.proximity_tolerance = proximity_tolerance

        # Your exact class weights
        self.class_weights = torch.tensor([
            0.1,  # Nearly ignore non-switch classes
            0.1,
            5.0,  # Focus on switches
            5.0
        ])

    def forward(self, logits, labels):
        """
        Same as your original but WITHOUT segmentation adjustment.
        No tokens parameter needed since we don't check for "/" or "།"
        """
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

        # NO SEGMENTATION ALIGNMENT CHECK HERE
        # This is the ONLY part removed from your original code

        # Your exact proximity logic (unchanged)
        for b in range(batch_size):
            true_switches = torch.where(labels[b] >= 2)[0]
            pred_switches = torch.where(predictions[b] >= 2)[0]

            # For each true switch, reward ANY prediction within tolerance
            for true_pos in true_switches:
                window_start = max(0, true_pos - self.proximity_tolerance)
                window_end = min(seq_len, true_pos + self.proximity_tolerance + 1)
                window_preds = predictions[b, window_start:window_end]
                if torch.any(window_preds >= 2):
                    base_loss[b, true_pos] *= 0.1  # Big reward

            # Mild penalty for predictions far from any true switch
            for pred_pos in pred_switches:
                if len(true_switches) > 0:
                    distances = torch.abs(true_switches - pred_pos)
                    min_distance = torch.min(distances).item()

                    if min_distance > self.proximity_tolerance:
                        base_loss[b, pred_pos] *= 2.0

        valid_mask = (labels != -100).float()
        return (base_loss * valid_mask).sum() / valid_mask.sum()


# ============================================================================
# MODIFIED TRAINER - NO SEGMENTATION ANALYSIS
# ============================================================================

class SimpleSwitchTrainerNoSegmentation(Trainer):
    """
    Your exact trainer but WITHOUT analyze_segmentation_marks.
    Still has logical constraints, just no segmentation detection.
    """

    def __init__(self, *args, **kwargs):
        # Store tokenizer reference
        self.tokenizer = kwargs.get('processing_class') or kwargs.get('tokenizer')

        # Handle tokenizer/processing_class conflict
        if 'tokenizer' in kwargs and 'processing_class' in kwargs:
            kwargs.pop('tokenizer')
        elif 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')

        super().__init__(*args, **kwargs)

        # Loss WITHOUT segmentation parameters
        self.loss_fn = SwitchFocusedLossNoSegmentation(
            switch_recall_weight=10.0,
            proximity_tolerance=5
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Apply logical constraints (keeping this part)
        logits = self.apply_logical_constraints(logits, labels)

        # NO analyze_segmentation_marks() call here
        # Just compute loss without segmentation info
        loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def apply_logical_constraints(self, logits, labels):
        """Your EXACT logical transition constraints (unchanged)."""
        batch_size, seq_len, _ = logits.shape

        for b in range(batch_size):
            prev_pred = None
            for t in range(seq_len):
                if labels[b, t] == -100:
                    continue

                current_pred = torch.argmax(logits[b, t])

                if prev_pred is not None:
                    # Auto (0) cannot transition to Switch-to-Auto (2)
                    if prev_pred == 0 and current_pred == 2:
                        logits[b, t, 2] = -float('inf')

                    # Allo (1) cannot transition to Switch-to-Allo (3)
                    if prev_pred == 1 and current_pred == 3:
                        logits[b, t, 3] = -float('inf')

                prev_pred = current_pred.item()

        return logits


# ============================================================================
# YOUR EXACT METRICS (unchanged)
# ============================================================================

def compute_metrics_for_trainer(eval_pred, tolerance=5):
    """Your exact metrics computation with proximity tolerance."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Flatten and filter
    true_labels = []
    pred_labels = []

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] != -100:
                true_labels.append(labels[i][j])
                pred_labels.append(predictions[i][j])

    # Calculate accuracy
    accuracy = np.mean([1 if t == p else 0 for t, p in zip(true_labels, pred_labels)])

    # Switch detection with tolerance
    true_switches = [i for i, l in enumerate(true_labels) if l in [2, 3]]
    pred_switches = [i for i, l in enumerate(pred_labels) if l in [2, 3]]

    # Count matches with tolerance
    matched = 0
    for true_pos in true_switches:
        for pred_pos in pred_switches:
            if abs(true_pos - pred_pos) <= tolerance:
                matched += 1
                break

    precision = matched / len(pred_switches) if pred_switches else 0.0
    recall = matched / len(true_switches) if true_switches else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'switch_f1': f1,
        'switch_precision': precision,
        'switch_recall': recall,
        'true_switches': len(true_switches),
        'pred_switches': len(pred_switches),
    }


# ============================================================================
# TRAINING FUNCTION FOR SINGLE MODEL
# ============================================================================

def train_single_model(model_name: str, save_name: str):
    """Train a single model with ALTO architecture minus segmentation."""

    print(f"\n{'=' * 80}")
    print(f"TRAINING: {model_name}")
    print(f"{'=' * 80}")
    print(f"Configuration: ALTO WITHOUT segmentation marks")
    print(f"Output: {save_name}")

    output_dir = f'{OUTPUT_BASE_DIR}/{save_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Data paths
    train_path = f'{DATA_DIR}/train_segments_with_more_auto_allo.csv'
    val_path = f'{DATA_DIR}/val_segments_more_auto_allo.csv'
    test_path = f'{DATA_DIR}/test_segments_original.csv'

    try:
        # Initialize model
        print(f"\nLoading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=4,
            ignore_mismatched_sizes=True
        )

        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✅ Model on GPU")

        # Create datasets
        print("Loading datasets...")
        train_dataset = CodeSwitchingDataset4Class(train_path, tokenizer)
        val_dataset = CodeSwitchingDataset4Class(val_path, tokenizer)
        test_dataset = CodeSwitchingDataset4Class(test_path, tokenizer)

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            max_length=512
        )

        # YOUR EXACT training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=30,
            save_strategy="steps",
            save_steps=60,
            learning_rate=2e-5,  # Your exact value
            per_device_train_batch_size=8,  # Your exact value
            per_device_eval_batch_size=8,  # Your exact value
            num_train_epochs=10,  # Your exact value
            weight_decay=0.01,
            logging_steps=20,
            load_best_model_at_end=True,
            metric_for_best_model='switch_f1',
            greater_is_better=True,
            warmup_steps=100,  # Your exact value
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            report_to=[],
            gradient_accumulation_steps=1,
            label_smoothing_factor=0.0
        )

        # Initialize trainer WITHOUT segmentation
        trainer = SimpleSwitchTrainerNoSegmentation(
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

        # Train
        print("\n" + "=" * 60)
        print("Training with ALTO (NO segmentation marks):")
        print(f"  ✅ 10 epochs")
        print(f"  ✅ Learning rate: 2e-5")
        print(f"  ✅ Batch size: 8")
        print(f"  ✅ Switch recall weight: 10.0")
        print(f"  ✅ Proximity tolerance: 5 tokens")
        print(f"  ❌ NO segmentation penalty/reward")
        print(f"  ✅ Logical constraints: Auto→Allo, Allo→Auto only")
        print("=" * 60)

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Save model
        trainer.save_model(f'{output_dir}/final_model')
        tokenizer.save_pretrained(f'{output_dir}/final_model')

        print(f"\n✅ Training complete in {training_time / 60:.1f} minutes")
        print(f"   Model saved to: {output_dir}/final_model")

        return True, {'training_time': training_time, 'status': 'success'}

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False, {'error': str(e), 'status': 'failed'}

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Train all models with ALTO minus segmentation marks."""

    print("=" * 80)
    print("MULTI-MODEL ALTO TRAINING (WITHOUT SEGMENTATION MARKS)")
    print("=" * 80)
    print("This is your EXACT ALTO architecture with ONLY")
    print("the '/' and '།' detection removed")
    print("\nKept:")
    print("  ✅ Proximity-aware loss")
    print("  ✅ Logical constraints")
    print("  ✅ Same class weights")
    print("  ✅ Same training parameters")
    print("\nRemoved:")
    print("  ❌ Segmentation mark detection")
    print("  ❌ Penalty for false switches at '/' or '།'")
    print("  ❌ Reward for correct non-switches at boundaries")

    print(f"\nModels to train: {len(MODELS_CONFIG)}")
    for model, save_name in MODELS_CONFIG.items():
        print(f"  • {model} → {save_name}")

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    results = {}
    successful = []
    failed = []

    for i, (model_name, save_name) in enumerate(MODELS_CONFIG.items(), 1):
        print(f"\n{'=' * 80}")
        print(f"MODEL {i}/{len(MODELS_CONFIG)}")
        print(f"{'=' * 80}")

        success, info = train_single_model(model_name, save_name)
        results[model_name] = info

        if success:
            successful.append(model_name)
        else:
            failed.append(model_name)

        # Save progress
        with open(f'{OUTPUT_BASE_DIR}/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(f"\n✅ Successful: {len(successful)}/{len(MODELS_CONFIG)}")
    for model in successful:
        print(f"   • {MODELS_CONFIG[model]}")
        print(f"     Time: {results[model]['training_time'] / 60:.1f} min")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for model in failed:
            print(f"   • {model}")

    print(f"\n📁 Models saved to: {OUTPUT_BASE_DIR}/")


if __name__ == "__main__":
    main()