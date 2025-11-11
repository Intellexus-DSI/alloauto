"""
Multi-Model Training with EXACT ALTO Architecture Configuration
================================================================
Uses YOUR EXACT parameters from the code:
- 10 epochs (not 30)
- Learning rate: 2e-5
- Batch size: 8
- Eval steps: 30
- Save steps: 60
- Warmup steps: 100
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change if needed
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================================
# MODELS TO TRAIN
# ============================================================================

# Dictionary mapping model names to saved directory names
MODELS_TO_TRAIN = {
    # 'hfl/cino-base-v2': 'cino_base_v2_ALTO_arch_23_10',
    # 'bert-base-multilingual-cased': 'mbert_cased_ALTO_arch_23_10',
    # 'xlm-roberta-base': 'xlm_roberta_base_ALTO_arch_23_10',
    # 'sangjeedondrub/tibetan-roberta-base': 'tibetan_roberta_ALTO_arch_23_10',
    # 'OMRIDRORI/mbert-tibetan-continual-wylie-final': 'mbert_tibetan_wylie_ALTO_arch_23_10',
    'OMRIDRORI/mbert-tibetan-continual-wylie-final': 'mbert_tibetan_wylie_ALTO_arch_24_10',

}


# Directories
DATA_DIR = 'dataset/preprocessed_augmented'  # Your augmented data
# for ALTO fine tune and save:
OUTPUT_BASE_DIR = './alloauto-segmentation-training/fine_tuned_ALTO_models'

#TODO -> if want to fine tune other backbones -> unmask this :
# for Benchmarks fine tune and save:
# OUTPUT_BASE_DIR = './alloauto-segmentation-training/benchmark_models_ALTO_architecture'

# ============================================================================
# YOUR EXACT DATASET CLASS
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
# YOUR EXACT LOSS FUNCTION
# ============================================================================

class SwitchFocusedLoss(nn.Module):
    """Your exact loss function with segmentation penalties/rewards."""

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

        # Apply proximity-based weighting for switches
        for b in range(batch_size):
            # Find true switch positions
            switch_positions = torch.where((labels[b] == 2) | (labels[b] == 3))[0]

            for switch_pos in switch_positions:
                # Apply proximity tolerance
                for offset in range(-self.proximity_tolerance, self.proximity_tolerance + 1):
                    pos = switch_pos + offset
                    if 0 <= pos < seq_len and valid_mask[b, pos]:
                        pred = torch.argmax(logits[b, pos])

                        if offset == 0:
                            # Exact position
                            if pred in [2, 3]:
                                weights[b, pos] = self.switch_recall_weight
                            else:
                                weights[b, pos] = self.switch_recall_weight * 2
                        else:
                            # Within tolerance
                            distance_factor = 1.0 - (abs(offset) / (self.proximity_tolerance + 1))
                            if pred in [2, 3]:
                                weights[b, pos] *= (1.0 + distance_factor)

        # Apply segmentation marks penalty/reward
        if seg_marks is not None:
            for b in range(batch_size):
                for t in range(seq_len):
                    if seg_marks[b, t] > 0 and valid_mask[b, t]:
                        pred = torch.argmax(logits[b, t])
                        true_label = labels[b, t]

                        # Penalty for predicting switch at segmentation mark
                        if pred in [2, 3] and true_label not in [2, 3]:
                            weights[b, t] *= self.segmentation_penalty
                        # Reward for NOT predicting switch at segmentation mark
                        elif pred not in [2, 3] and true_label not in [2, 3]:
                            weights[b, t] *= self.segmentation_reward

        # Apply weights and compute final loss
        weighted_loss = ce_loss * weights
        total_loss = weighted_loss[valid_mask].mean()

        return total_loss

# ============================================================================
# YOUR EXACT TRAINER CLASS
# ============================================================================

class SimpleSwitchTrainer(Trainer):
    """Your exact trainer with logical constraints and segmentation analysis."""

    def __init__(self, *args, **kwargs):
        # Store tokenizer reference before modifying kwargs
        self.tokenizer = kwargs.get('processing_class') or kwargs.get('tokenizer')

        # Remove tokenizer from kwargs if both are present to avoid conflict
        if 'tokenizer' in kwargs and 'processing_class' in kwargs:
            kwargs.pop('tokenizer')
        elif 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')

        super().__init__(*args, **kwargs)

        # Your exact loss configuration
        self.loss_fn = SwitchFocusedLoss(
            switch_recall_weight=10.0,
            proximity_tolerance=5,
            segmentation_penalty=3.0,
            segmentation_reward=0.3
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_ids = inputs.get("input_ids").clone()
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Apply logical constraints
        logits = self.apply_logical_constraints(logits, labels)

        # Analyze segmentation marks
        tokens_with_seg_info = self.analyze_segmentation_marks(input_ids)

        # Compute loss
        loss = self.loss_fn(logits, labels, tokens_with_seg_info)

        return (loss, outputs) if return_outputs else loss

    def apply_logical_constraints(self, logits, labels):
        """Apply your exact logical transition constraints."""
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

    def analyze_segmentation_marks(self, input_ids):
        """Identify segmentation marks ('/' and '།') in tokens."""
        batch_size, seq_len = input_ids.shape
        seg_marks = torch.zeros_like(input_ids, dtype=torch.float)

        for b in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b])
            for t, token in enumerate(tokens):
                if token and ('/' in str(token) or '།' in str(token)):
                    seg_marks[b, t] = 1.0

        return seg_marks

# ============================================================================
# METRICS COMPUTATION (YOUR EXACT METRICS)
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
# TRAIN SINGLE MODEL FUNCTION
# ============================================================================

def train_single_model(model_name: str, output_dir: str):
    """Train a single model with YOUR EXACT configuration."""

    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Data paths - matching your actual file names
    train_path = f'{DATA_DIR}/train_segments_with_more_auto_allo.csv'
    val_path = f'{DATA_DIR}/val_segments_more_auto_allo.csv'
    test_path = f'{DATA_DIR}/test_segments_original.csv'

    try:
        # Initialize tokenizer and model
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=4,
            ignore_mismatched_sizes=True
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✅ Model on GPU: {torch.cuda.get_device_name(0)}")

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
            eval_steps=30,                      # Your exact value
            save_strategy="steps",
            save_steps=60,                       # Your exact value
            learning_rate=2e-5,                  # Your exact value
            per_device_train_batch_size=8,      # Your exact value
            per_device_eval_batch_size=8,       # Your exact value
            num_train_epochs=10,                 # YOUR EXACT VALUE: 10 epochs
            weight_decay=0.01,
            logging_steps=20,                    # Your exact value
            load_best_model_at_end=True,
            metric_for_best_model='switch_f1',
            greater_is_better=True,
            warmup_steps=100,                    # Your exact value
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to=[],
            gradient_accumulation_steps=1,
            label_smoothing_factor=0.0,
            push_to_hub=True,  # Enable pushing to HF
            hub_model_id="levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_MUL_SEG_RUNI",  # Your HF repo

        )

        # Initialize YOUR EXACT trainer
        trainer = SimpleSwitchTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,  # Only pass as processing_class
            compute_metrics=lambda eval_pred: compute_metrics_for_trainer(eval_pred, tolerance=5)
        )

        # Add early stopping (patience=5 from your code)
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

        # Train
        print("\n" + "="*60)
        print("Training with YOUR EXACT configuration:")
        print(f"  • 10 epochs")
        print(f"  • Learning rate: 2e-5")
        print(f"  • Batch size: 8")
        print(f"  • Switch recall weight: 10.0")
        print(f"  • Proximity tolerance: 5 tokens")
        print(f"  • Segmentation penalty: 3.0")
        print(f"  • Segmentation reward: 0.3")
        print(f"  • Logical constraints: Auto→Allo, Allo→Auto only")
        print("="*60)

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Save final model
        trainer.save_model(f'{output_dir}/final_model')
        tokenizer.save_pretrained(f'{output_dir}/final_model')

        print(f"\n✅ Training complete in {training_time/60:.1f} minutes")
        print(f"   Model saved to: {output_dir}/final_model")

        return True, {'training_time': training_time, 'status': 'success'}

    except Exception as e:
        print(f"\n❌ Error training {model_name}: {str(e)}")
        return False, {'error': str(e), 'status': 'failed'}

    finally:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Train all models with YOUR EXACT ALTO configuration."""

    print("="*80)
    print("MULTI-MODEL ALTO TRAINING")
    print("WITH YOUR EXACT CONFIGURATION")
    print("="*80)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU available")

    print(f"\nModels to train: {len(MODELS_TO_TRAIN)}")
    for i, (model_name, save_name) in enumerate(MODELS_TO_TRAIN.items(), 1):
        print(f"  {i}. {model_name}")
        print(f"     → Will be saved as: {save_name}")

    # Create base directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Track results
    results = {}

    # Train each model
    for i, (model_name, save_name) in enumerate(MODELS_TO_TRAIN.items(), 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(MODELS_TO_TRAIN)}: {save_name}")
        print(f"{'='*80}")

        # Create model-specific directory with clean name
        model_output_dir = f'{OUTPUT_BASE_DIR}/{save_name}'

        # Train
        success, info = train_single_model(model_name, model_output_dir)
        results[model_name] = {
            **info,
            'save_name': save_name,
            'output_dir': model_output_dir
        }

        # Save intermediate results
        with open(f'{OUTPUT_BASE_DIR}/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    successful = [m for m, r in results.items() if r.get('status') == 'success']
    failed = [m for m, r in results.items() if r.get('status') == 'failed']

    print(f"\n✅ Successful: {len(successful)}/{len(MODELS_TO_TRAIN)}")
    for model in successful:
        time_min = results[model]['training_time'] / 60
        save_name = results[model]['save_name']
        print(f"   • {save_name}: {time_min:.1f} min")
        print(f"     ({model})")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for model in failed:
            save_name = results[model]['save_name']
            print(f"   • {save_name} ({model})")

    print(f"\n📁 All models saved to: {OUTPUT_BASE_DIR}/")
    print("   Directory structure:")
    for model_name, save_name in MODELS_TO_TRAIN.items():
        print(f"   └── {save_name}/")
        print(f"       └── final_model/")

    print(f"\n📊 Results saved to: {OUTPUT_BASE_DIR}/training_results.json")

if __name__ == "__main__":
    main()