"""
XLM-RoBERTa baseline - uses EXACT SAME data splits as ALTO model
XLM-RoBERTa is often stronger than mBERT for multilingual tasks
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
# Same dataset class
# ============================================================================

class CodeSwitchingDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        print(f"Loading data from: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

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
# Weighted trainer
# ============================================================================

class WeightedTrainer(Trainer):
    def __init__(self, switch_weight=30.0, *args, **kwargs):
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.get('tokenizer')
        super().__init__(*args, **kwargs)

        self.class_weights = torch.tensor([
            1.0, 1.0, switch_weight, switch_weight
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
# Metrics
# ============================================================================

import numpy as np


def evaluate_with_proximity(true_labels, pred_labels, tolerance=5):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    true_auto = np.where(true_labels == 2)[0]
    true_allo = np.where(true_labels == 3)[0]
    pred_auto = np.where(pred_labels == 2)[0]
    pred_allo = np.where(pred_labels == 3)[0]

    matched_true = set()

    for pred_pos in pred_auto:
        if len(true_auto) > 0:
            distances = np.abs(true_auto - pred_pos)
            min_dist = np.min(distances)
            closest_true = true_auto[np.argmin(distances)]
            if closest_true not in matched_true and min_dist <= tolerance:
                matched_true.add(closest_true)

    for pred_pos in pred_allo:
        if len(true_allo) > 0:
            distances = np.abs(true_allo - pred_pos)
            min_dist = np.min(distances)
            closest_true = true_allo[np.argmin(distances)]
            if closest_true not in matched_true and min_dist <= tolerance:
                matched_true.add(closest_true)

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

    prox = evaluate_with_proximity(all_labels, all_preds, tolerance=5)

    return {
        'precision': float(prox['precision']),
        'recall': float(prox['recall']),
        'f1': float(prox['f1'])
    }


# ============================================================================
# Main training
# ============================================================================

def train_tibetan_roberta_baseline():
    print("=" * 80)
    print("TRAINING XLM-RoBERTa BASELINE")
    print("Using EXACT SAME data splits as ALTO model")
    print("=" * 80)

    # Check for data files
    if os.path.exists('train_segments_clean.csv'):
        print("\n✓ Found stratified splits")
        TRAIN_FILE = 'train_segments_clean.csv'
        VAL_FILE = 'val_segments.csv'
        TEST_FILE = 'test_segments.csv'
    # elif os.path.exists('train_segments.csv'):
    #     print("\n✓ Found regular splits")
    #     TRAIN_FILE = 'train_segments.csv'
    #     VAL_FILE = 'val_segments.csv'
    #     TEST_FILE = 'test_segments.csv'
    else:
        print("\n❌ ERROR: No data splits found!")
        return None, None, None
    OUTPUT_DIR = './alloauto-segmentation-training/benchmark_models/tibetan-roberta_baseline_model_clean_train'


    # Load XLM-RoBERTa
    model_name = 'sangjeedondrub/tibetan-roberta-base'
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        add_prefix_space=True  # <-- REQUIRED for RoBERTa when is_split_into_words=True
    )
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

    # Load datasets
    print(f"\nLoading datasets:")
    train_dataset = CodeSwitchingDataset(TRAIN_FILE, tokenizer)
    val_dataset = CodeSwitchingDataset(VAL_FILE, tokenizer)
    test_dataset = CodeSwitchingDataset(TEST_FILE, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=60,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=15,
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

    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        switch_weight=30.0
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    trainer.save_model(f'{OUTPUT_DIR}/final_model')
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/final_model')

    print(f"\n✅ Model saved to: {OUTPUT_DIR}/final_model")

    # Evaluate
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\n=== XLM-RoBERTa Test Results ===")
    print(f"Precision: {test_results['eval_precision']:.3f}")
    print(f"Recall: {test_results['eval_recall']:.3f}")
    print(f"F1: {test_results['eval_f1']:.3f}")

    return trainer, model, tokenizer


if __name__ == "__main__":
    trainer, model, tokenizer = train_tibetan_roberta_baseline()