"""
Multi-Model Training with EXACT ALTO Architecture Configuration - 3-CLASS VERSION
==================================================================================
Converts 4-class to 3-class:
- Class 0: Non-switch (was non_switch_auto + non_switch_allo)
- Class 1: Switch→Auto
- Class 2: Switch→Allo

Uses YOUR EXACT parameters:
- 10 epochs
- Learning rate: 2e-5
- Batch size: 8
- Switch recall weight: 10.0
- Proximity tolerance: 5
- Segmentation penalty: 3.0
- Segmentation reward: 0.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import ssl
import gc
import time
from datetime import datetime
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change if needed
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================================
# MODELS TO TRAIN
# ============================================================================

MODELS_TO_TRAIN = {
    # 'hfl/cino-base-v2': 'cino_base_v2_ALTO_arch_3class_23_10',
    # 'bert-base-multilingual-cased': 'mbert_cased_ALTO_MUL_SEG_arch_3class_24_10',
    # 'xlm-roberta-base': 'xlm_roberta_base_ALTO_arch_3class_23_10',
    # 'sangjeedondrub/tibetan-roberta-base': 'tibetan_roberta_ALTO_arch_3class_23_10',
    'OMRIDRORI/mbert-tibetan-continual-wylie-final': 'mbert_tibetan_wylie_ALTO_arch_3class_24_10',
}

# Directories
DATA_DIR = 'dataset/preprocessed_augmented'
# OUTPUT_BASE_DIR = './alloauto-segmentation-training/fine_tuned_ALTO_models_3class'
OUTPUT_BASE_DIR = './alloauto-segmentation-training/benchmark_models_ALTO_architecture'



# ============================================================================
# LABEL REMAPPING FUNCTION
# ============================================================================

def remap_4class_to_3class(label_4class):
    """
    Remap 4-class labels to 3-class:
    4-class: 0=non_switch_auto, 1=non_switch_allo, 2=switch→auto, 3=switch→allo
    3-class: 0=non_switch, 1=switch→auto, 2=switch→allo
    """
    if label_4class == -100:
        return -100
    elif label_4class in [0, 1]:  # Both non-switch types → class 0
        return 0
    elif label_4class == 2:  # Switch to auto → class 1
        return 1
    elif label_4class == 3:  # Switch to allo → class 2
        return 2
    else:
        return 0  # Default to non-switch


# ============================================================================
# 3-CLASS DATASET
# ============================================================================

class CodeSwitchingDataset3Class(Dataset):
    """3-class dataset - remaps 4-class labels to 3-class on the fly."""

    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"  Loaded {len(self.data)} segments from {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get tokens and 4-class labels
        tokens = row['tokens'].split()
        labels_4class = [int(l) for l in row['labels'].split(',')]

        # Remap to 3-class
        labels_3class = [remap_4class_to_3class(l) for l in labels_4class]

        # Ensure alignment
        if len(tokens) != len(labels_3class):
            min_len = min(len(tokens), len(labels_3class))
            tokens = tokens[:min_len]
            labels_3class = labels_3class[:min_len]

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
                if word_idx < len(labels_3class):
                    aligned_labels.append(labels_3class[word_idx])
                else:
                    aligned_labels.append(-100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        encoding["labels"] = torch.tensor(aligned_labels)

        return {key: val.squeeze() for key, val in encoding.items()}


# ============================================================================
# 3-CLASS LOSS FUNCTION
# ============================================================================

class SwitchFocusedLoss3Class(nn.Module):
    """3-class version of your ALTO loss function."""

    def __init__(self, switch_recall_weight=10.0, proximity_tolerance=5,
                 segmentation_penalty=3.0, segmentation_reward=0.3):
        super().__init__()
        self.switch_recall_weight = switch_recall_weight
        self.proximity_tolerance = proximity_tolerance
        self.segmentation_penalty = segmentation_penalty
        self.segmentation_reward = segmentation_reward

    def forward(self, logits, labels, seg_marks=None):
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device

        # Create mask for valid labels
        valid_mask = labels != -100

        # Base cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        ce_loss = ce_loss.view(batch_size, seq_len)

        # Initialize weights
        weights = torch.ones_like(ce_loss)

        # Apply proximity-based weighting for switches (classes 1 and 2)
        for b in range(batch_size):
            # Find true switch positions (classes 1 and 2)
            switch_positions = torch.where((labels[b] == 1) | (labels[b] == 2))[0]

            for switch_pos in switch_positions:
                # Apply proximity tolerance
                for offset in range(-self.proximity_tolerance, self.proximity_tolerance + 1):
                    pos = switch_pos + offset
                    if 0 <= pos < seq_len and valid_mask[b, pos]:
                        pred = torch.argmax(logits[b, pos])

                        if offset == 0:
                            # Exact position
                            if pred in [1, 2]:  # Predicted a switch
                                weights[b, pos] = self.switch_recall_weight
                            else:
                                weights[b, pos] = self.switch_recall_weight * 2
                        else:
                            # Within tolerance
                            distance_factor = 1.0 - (abs(offset) / (self.proximity_tolerance + 1))
                            if pred in [1, 2]:  # Predicted a switch
                                weights[b, pos] *= (1.0 + distance_factor)

        # Apply segmentation marks penalty/reward
        if seg_marks is not None:
            for b in range(batch_size):
                for t in range(seq_len):
                    if seg_marks[b, t] > 0 and valid_mask[b, t]:
                        pred = torch.argmax(logits[b, t])
                        true_label = labels[b, t]

                        # Penalty for predicting switch at segmentation mark
                        if pred in [1, 2] and true_label not in [1, 2]:
                            weights[b, t] *= self.segmentation_penalty
                        # Reward for NOT predicting switch at segmentation mark
                        elif pred not in [1, 2] and true_label not in [1, 2]:
                            weights[b, t] *= self.segmentation_reward

        # Apply weights and compute final loss
        weighted_loss = ce_loss * weights
        total_loss = weighted_loss[valid_mask].mean()

        return total_loss


# ============================================================================
# 3-CLASS TRAINER
# ============================================================================

class SimpleSwitchTrainer3Class(Trainer):
    """3-class version of your ALTO trainer."""

    def __init__(self, *args, **kwargs):
        # Store tokenizer reference
        self.tokenizer = kwargs.get('processing_class') or kwargs.get('tokenizer')

        # Remove tokenizer from kwargs if both are present
        if 'tokenizer' in kwargs and 'processing_class' in kwargs:
            kwargs.pop('tokenizer')

        # Extract custom parameters
        self.switch_recall_weight = kwargs.pop('switch_recall_weight', 10.0)
        self.proximity_tolerance = kwargs.pop('proximity_tolerance', 5)
        self.segmentation_penalty = kwargs.pop('segmentation_penalty', 3.0)
        self.segmentation_reward = kwargs.pop('segmentation_reward', 0.3)
        self.apply_constraints = kwargs.pop('apply_constraints', True)

        super().__init__(*args, **kwargs)

        # Initialize custom loss
        self.custom_loss = SwitchFocusedLoss3Class(
            switch_recall_weight=self.switch_recall_weight,
            proximity_tolerance=self.proximity_tolerance,
            segmentation_penalty=self.segmentation_penalty,
            segmentation_reward=self.segmentation_reward
        )

        print(f"✓ 3-Class Trainer initialized")
        print(f"  Switch recall weight: {self.switch_recall_weight}")
        print(f"  Proximity tolerance: {self.proximity_tolerance}")
        print(f"  Segmentation penalty: {self.segmentation_penalty}")
        print(f"  Segmentation reward: {self.segmentation_reward}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation for 3-class."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Get segmentation marks if available
        seg_marks = inputs.get("seg_marks", None)

        # Compute custom loss
        loss = self.custom_loss(logits, labels, seg_marks)

        return (loss, outputs) if return_outputs else loss

    def apply_logical_constraints_3class(self, predictions):
        """
        Apply logical constraints for 3-class:
        - Switch→Auto (1) should follow non-switch in allo mode
        - Switch→Allo (2) should follow non-switch in auto mode
        """
        predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        predictions = predictions.copy()

        current_mode = 0  # Start in auto mode

        for i in range(len(predictions)):
            pred = predictions[i]

            if pred == -100:
                continue

            # Determine if this is a valid switch
            if pred == 1:  # Switch to auto
                if current_mode != 1:  # Not in allo mode
                    predictions[i] = 0  # Change to non-switch
                else:
                    current_mode = 0  # Now in auto mode

            elif pred == 2:  # Switch to allo
                if current_mode != 0:  # Not in auto mode
                    predictions[i] = 0  # Change to non-switch
                else:
                    current_mode = 1  # Now in allo mode

            # Track current mode from non-switch predictions
            # In 3-class, we need to infer mode from context
            elif pred == 0:
                # Mode stays as is (we can't distinguish auto/allo in non-switch)
                pass

        return predictions

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Prediction with optional constraints."""
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            loss = self.compute_loss(model, inputs, return_outputs=False)

        predictions = torch.argmax(logits, dim=-1)

        if self.apply_constraints:
            batch_size = predictions.shape[0]
            for b in range(batch_size):
                predictions[b] = torch.tensor(
                    self.apply_logical_constraints_3class(predictions[b]),
                    device=predictions.device
                )

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)


# ============================================================================
# 3-CLASS METRICS
# ============================================================================

def compute_metrics_for_trainer_3class(eval_pred, tolerance=5):
    """Compute metrics for 3-class with proximity tolerance."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Flatten and filter padding
    flat_preds = []
    flat_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                flat_preds.append(pred)
                flat_labels.append(label)

    flat_preds = np.array(flat_preds)
    flat_labels = np.array(flat_labels)

    # Basic accuracy
    accuracy = np.mean(flat_preds == flat_labels)

    # Switch detection with proximity (classes 1 and 2 are switches)
    true_switches = [i for i, l in enumerate(flat_labels) if l in [1, 2]]
    pred_switches = [i for i, l in enumerate(flat_preds) if l in [1, 2]]

    # ✅ CHANGED: Use same logic as 4-class
    matched = 0
    for true_pos in true_switches:
        for pred_pos in pred_switches:
            if abs(true_pos - pred_pos) <= tolerance:
                matched += 1
                break  # ← Prevents double-counting like 4-class

    recall = matched / len(true_switches) if len(true_switches) > 0 else 0.0
    precision = matched / len(pred_switches) if len(pred_switches) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'switch_f1': f1,
        'switch_precision': precision,
        'switch_recall': recall,
        'true_switches': len(true_switches),
        'pred_switches': len(pred_switches),
    }

# ============================================================================
# TRAIN SINGLE MODEL
# ============================================================================

def train_single_model_3class(model_name: str, output_dir: str):
    """Train a single 3-class model with ALTO architecture."""

    print(f"\n{'=' * 80}")
    print(f"TRAINING 3-CLASS: {model_name}")
    print(f"{'=' * 80}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Data paths
    train_path = f'{DATA_DIR}/train_segments_with_more_auto_allo.csv'
    val_path = f'{DATA_DIR}/val_segments_more_auto_allo.csv'
    test_path = f'{DATA_DIR}/test_segments_original.csv'

    try:
        # Initialize tokenizer and model
        print(f"Loading {model_name}...")

        # Special handling for RoBERTa models
        if 'roberta' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                add_prefix_space=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 3-class model
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=3,  # ← 3 classes instead of 4
            label2id={'non_switch': 0, 'to_auto': 1, 'to_allo': 2},
            id2label={0: 'non_switch', 1: 'to_auto', 2: 'to_allo'},
            ignore_mismatched_sizes=True
        )

        # Move to GPU
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✅ Model on GPU: {torch.cuda.get_device_name(0)}")

        # Create 3-class datasets
        print("Loading datasets (4-class → 3-class remapping)...")
        train_dataset = CodeSwitchingDataset3Class(train_path, tokenizer)
        val_dataset = CodeSwitchingDataset3Class(val_path, tokenizer)
        test_dataset = CodeSwitchingDataset3Class(test_path, tokenizer)

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            max_length=512
        )

        # Training arguments (YOUR EXACT VALUES)
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
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to=[],
            gradient_accumulation_steps=1,
            label_smoothing_factor=0.0
        )

        # Initialize 3-class trainer
        trainer = SimpleSwitchTrainer3Class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_metrics=lambda eval_pred: compute_metrics_for_trainer_3class(eval_pred, tolerance=5),
            switch_recall_weight=10.0,
            proximity_tolerance=5,
            segmentation_penalty=3.0,
            segmentation_reward=0.3,
            apply_constraints=True
        )

        # Add early stopping
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

        # Train
        print("\n" + "=" * 60)
        print("Training 3-CLASS model with ALTO architecture:")
        print(f"  • 3 classes: Non-switch, Switch→Auto, Switch→Allo")
        print(f"  • 10 epochs")
        print(f"  • Learning rate: 2e-5")
        print(f"  • Batch size: 8")
        print(f"  • Switch recall weight: 10.0")
        print(f"  • Proximity tolerance: 5 tokens")
        print(f"  • Segmentation penalty: 3.0")
        print(f"  • Segmentation reward: 0.3")
        print(f"  • Logical constraints enabled")
        print("=" * 60)

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Save final model
        trainer.save_model(f'{output_dir}/final_model')
        tokenizer.save_pretrained(f'{output_dir}/final_model')

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)

        print(f"\n=== 3-Class Test Results ===")
        print(f"Accuracy: {test_results.get('eval_accuracy', 0):.3f}")
        print(f"Switch F1: {test_results.get('eval_switch_f1', 0):.3f}")
        print(f"Switch Precision: {test_results.get('eval_switch_precision', 0):.3f}")
        print(f"Switch Recall: {test_results.get('eval_switch_recall', 0):.3f}")
        print(f"Training time: {training_time / 60:.1f} minutes")

        # Save results
        results = {
            'model_name': model_name,
            'test_results': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                             for k, v in test_results.items()},
            'training_time_seconds': training_time,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        with open(f'{output_dir}/results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Training complete")
        print(f"   Model saved to: {output_dir}/final_model")

        return True, results

    except Exception as e:
        print(f"\n❌ Error training {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e), 'status': 'failed'}

    finally:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def verify_switch_preservation():
    """Verify that 4→3 remapping preserves all switches."""
    test_df = pd.read_csv('dataset/preprocessed_augmented/test_segments_original.csv')

    mismatches = 0
    for idx in range(len(test_df)):
        labels_4 = [int(l) for l in test_df.iloc[idx]['labels'].split(',')]
        labels_3 = [remap_4class_to_3class(l) for l in labels_4]

        # Count switches
        switches_4 = sum(1 for l in labels_4 if l in [2, 3])
        switches_3 = sum(1 for l in labels_3 if l in [1, 2])

        # Count by type
        auto_4 = sum(1 for l in labels_4 if l == 2)
        auto_3 = sum(1 for l in labels_3 if l == 1)
        allo_4 = sum(1 for l in labels_4 if l == 3)
        allo_3 = sum(1 for l in labels_3 if l == 2)

        if switches_4 != switches_3 or auto_4 != auto_3 or allo_4 != allo_3:
            mismatches += 1
            print(f"Segment {idx}: 4-class switches={switches_4}, 3-class switches={switches_3}")

    if mismatches == 0:
        print(f"✅ All {len(test_df)} segments verified - switches preserved!")
    else:
        print(f"❌ {mismatches} segments have mismatches!")

    return mismatches == 0


# Run this before training
def main():
    """Train all models with 3-class ALTO configuration."""
    verify_switch_preservation()

    print("=" * 80)
    print("MULTI-MODEL 3-CLASS ALTO TRAINING")
    print("=" * 80)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU available")

    print(f"\nModels to train: {len(MODELS_TO_TRAIN)}")
    for i, (model_name, save_name) in enumerate(MODELS_TO_TRAIN.items(), 1):
        print(f"  {i}. {model_name}")
        print(f"     → {save_name}")

    # Create base directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Track results
    results = {}

    # Train each model
    for i, (model_name, save_name) in enumerate(MODELS_TO_TRAIN.items(), 1):
        print(f"\n{'=' * 80}")
        print(f"MODEL {i}/{len(MODELS_TO_TRAIN)}: {save_name}")
        print(f"{'=' * 80}")

        model_output_dir = f'{OUTPUT_BASE_DIR}/{save_name}'

        success, info = train_single_model_3class(model_name, model_output_dir)
        results[model_name] = {
            **info,
            'save_name': save_name,
            'output_dir': model_output_dir
        }

        # Save intermediate results
        with open(f'{OUTPUT_BASE_DIR}/training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 80)
    print("3-CLASS TRAINING COMPLETE")
    print("=" * 80)

    successful = [m for m, r in results.items() if r.get('status') == 'success']
    failed = [m for m, r in results.items() if r.get('status') == 'failed']

    print(f"\n✅ Successful: {len(successful)}/{len(MODELS_TO_TRAIN)}")
    for model in successful:
        save_name = results[model]['save_name']
        time_min = results[model].get('training_time_seconds', 0) / 60
        test_f1 = results[model].get('test_results', {}).get('eval_switch_f1', 0)
        print(f"   • {save_name}: {time_min:.1f} min, F1={test_f1:.3f}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for model in failed:
            save_name = results[model]['save_name']
            print(f"   • {save_name}")

    print(f"\n📁 All models saved to: {OUTPUT_BASE_DIR}/")
    print(f"📊 Results saved to: {OUTPUT_BASE_DIR}/training_results.json")


if __name__ == "__main__":
    main()