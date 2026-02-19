"""
Simple mBERT baseline - uses EXACT SAME data splits as ALTO model
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from torch.utils.data import Dataset
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ============================================================================
# IMPORTANT: Use the EXACT SAME dataset class and files as main model
# ============================================================================

class CodeSwitchingDataset(Dataset):
    """Same dataset class as your main training"""

    def __init__(self, csv_file, tokenizer, max_length=512):
        print(f"Loading data from: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Verify data
        print(f"  Segments: {len(self.data)}")
        print(f"  Files: {self.data['source_file'].nunique()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens'].split()
        labels = list(map(int, row['labels'].split(',')))

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        word_ids = encoding.word_ids()
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

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(aligned_labels)
        }


# ============================================================================
# Simple weighted loss (same concept as your proximity loss but simpler)
# ============================================================================

class WeightedTrainer(Trainer):
    def __init__(self, switch_weight=30.0, *args, **kwargs):
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.get('tokenizer')
        super().__init__(*args, **kwargs)

        self.class_weights = torch.tensor([
            1.0,           # Non-switch Auto
            1.0,           # Non-switch Allo
            switch_weight, # Switch to Auto
            switch_weight  # Switch to Allo
        ])

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        device = logits.device
        self.class_weights = self.class_weights.to(device)

        batch_size, seq_len, num_classes = logits.shape

        loss = nn.functional.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='mean',
            ignore_index=-100
        )

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Use your exact same evaluation metrics
# ============================================================================

import numpy as np

def evaluate_with_proximity(true_labels, pred_labels, tolerance=5):
    """Simplified version of your proximity evaluation"""
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    true_auto = np.where(true_labels == 2)[0]
    true_allo = np.where(true_labels == 3)[0]
    pred_auto = np.where(pred_labels == 2)[0]
    pred_allo = np.where(pred_labels == 3)[0]

    matched_true = set()
    matched_pred = set()

    # Match with tolerance
    for pred_pos in pred_auto:
        if len(true_auto) > 0:
            distances = np.abs(true_auto - pred_pos)
            min_dist = np.min(distances)
            closest_true = true_auto[np.argmin(distances)]
            if closest_true not in matched_true and min_dist <= tolerance:
                matched_true.add(closest_true)
                matched_pred.add(pred_pos)

    for pred_pos in pred_allo:
        if len(true_allo) > 0:
            distances = np.abs(true_allo - pred_pos)
            min_dist = np.min(distances)
            closest_true = true_allo[np.argmin(distances)]
            if closest_true not in matched_true and min_dist <= tolerance:
                matched_true.add(closest_true)
                matched_pred.add(pred_pos)

    total_true = len(true_auto) + len(true_allo)
    total_pred = len(pred_auto) + len(pred_allo)
    total_matched = len(matched_true)

    precision = total_matched / total_pred if total_pred > 0 else 0
    recall = total_matched / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    all_preds = predictions.flatten()
    all_labels = labels.flatten()

    mask = all_labels != -100
    all_preds = all_preds[mask]
    all_labels = all_labels[mask]

    accuracy = (all_preds == all_labels).mean()
    prox = evaluate_with_proximity(all_labels, all_preds, tolerance=5)

    return {
        'accuracy': float(accuracy),
        'precision': float(prox['precision']),
        'recall': float(prox['recall']),
        'f1': float(prox['f1'])
    }


# ============================================================================
# Main training - USES YOUR EXACT DATA FILES
# ============================================================================

def train_mbert_baseline():
    print("=" * 80)
    print("TRAINING VANILLA mBERT BASELINE")
    print("Using EXACT SAME data splits as ALTO model")
    print("=" * 80)

    # ========================================================================
    # CRITICAL: Check which CSV files exist and use the right ones
    # ========================================================================

    # Check if stratified splits exist (created by your main training)
    if os.path.exists('train_segments_clean.csv'):
        print("\n✓ Found stratified splits (created by main training)")
        TRAIN_FILE = 'train_segments_clean.csv'
        VAL_FILE = 'val_segments.csv'
        TEST_FILE = 'test_segments.csv'
    # elif os.path.exists('train_segments.csv'):
    #     print("\n✓ Found regular splits (created by main training)")
    #     TRAIN_FILE = 'train_segments.csv'
    #     VAL_FILE = 'val_segments.csv'
    #     TEST_FILE = 'test_segments.csv'
    else:
        print("\n❌ ERROR: No data splits found!")
        print("Please run your main training script first to create the splits.")
        print("Expected files: train_segments.csv, val_segments.csv, test_segments.csv")
        return None, None, None

    # Verify files exist
    for filepath in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not os.path.exists(filepath):
            print(f"❌ ERROR: {filepath} not found!")
            return None, None, None

    OUTPUT_DIR = './alloauto-segmentation-training/benchmark_models/mbert_baseline_model_clean_train'

    # ========================================================================
    # Load model
    # ========================================================================

    model_name = 'bert-base-multilingual-cased'
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=4,
        label2id={'non_switch_auto': 0, 'non_switch_allo': 1,
                  'to_auto': 2, 'to_allo': 3},
        id2label={0: 'non_switch_auto', 1: 'non_switch_allo',
                  2: 'to_auto', 3: 'to_allo'}
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # ========================================================================
    # Load EXACT SAME data splits
    # ========================================================================

    print(f"\nLoading datasets from EXACT same splits as ALTO model:")
    train_dataset = CodeSwitchingDataset(TRAIN_FILE, tokenizer)
    val_dataset = CodeSwitchingDataset(VAL_FILE, tokenizer)
    test_dataset = CodeSwitchingDataset(TEST_FILE, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # ========================================================================
    # Training arguments (simpler than your main model)
    # ========================================================================

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=60,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=15,  # Fewer epochs than ALTO
        weight_decay=0.1,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        warmup_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    # ========================================================================
    # Trainer with simple weighted loss
    # ========================================================================

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        switch_weight=30.0  # Same as your ALTO model
    )

    # ========================================================================
    # Train
    # ========================================================================

    print("\nStarting training...")
    trainer.train()

    # Save
    trainer.save_model(f'{OUTPUT_DIR}/final_model')
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/final_model')

    print(f"\n✅ Model saved to: {OUTPUT_DIR}/final_model")

    # ========================================================================
    # Evaluate on SAME test set
    # ========================================================================

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\n=== mBERT Baseline Test Results ===")
    print(f"Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Precision: {test_results['eval_precision']:.3f}")
    print(f"Recall: {test_results['eval_recall']:.3f}")
    print(f"F1: {test_results['eval_f1']:.3f}")

    return trainer, model, tokenizer


if __name__ == "__main__":
    trainer, model, tokenizer = train_mbert_baseline()