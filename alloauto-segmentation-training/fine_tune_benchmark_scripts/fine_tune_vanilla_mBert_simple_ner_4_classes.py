"""
Simple 4-Class Token Classification for Tibetan Text
Classes:
  0: Non-switching Auto
  1: Non-switching Allo
  2: Switch TO Auto
  3: Switch TO Allo

Straightforward approach without complex switching logic
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from datasets import load_metric

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change GPU if needed

print(f"Number of GPUs available: {torch.cuda.device_count()}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


# ============================================================================
# DATASET CLASS
# ============================================================================

class SimpleCodeSwitchingDataset(Dataset):
    """Simple dataset for 4-class token-level classification."""

    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loaded {len(self.data)} segments from {csv_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens'].split()
        labels = list(map(int, row['labels'].split(',')))

        encoding = self.tokenize_and_align_labels(tokens, labels)

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(encoding['labels'])
        }

    def tokenize_and_align_labels(self, tokens, labels):
        """Tokenize and align labels with subword tokens."""
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
                # Special tokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the label
                aligned_labels.append(labels[word_idx] if word_idx < len(labels) else -100)
            else:
                # Subsequent subwords of same word get -100
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        tokenized_inputs['labels'] = aligned_labels
        return tokenized_inputs


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """Compute standard classification metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Flatten and remove padding
    true_labels = []
    pred_labels = []

    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if labels[i][j] != -100:
                true_labels.append(labels[i][j])
                pred_labels.append(predictions[i][j])

    # Overall metrics
    accuracy = np.mean(np.array(pred_labels) == np.array(true_labels))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, labels=[0, 1, 2, 3], zero_division=0
    )

    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Switch detection metrics (classes 2 and 3 combined)
    switch_mask_true = np.array([l in [2, 3] for l in true_labels])
    switch_mask_pred = np.array([l in [2, 3] for l in pred_labels])

    switch_tp = np.sum(switch_mask_true & switch_mask_pred)
    switch_fp = np.sum(~switch_mask_true & switch_mask_pred)
    switch_fn = np.sum(switch_mask_true & ~switch_mask_pred)

    switch_precision = switch_tp / (switch_tp + switch_fp) if (switch_tp + switch_fp) > 0 else 0
    switch_recall = switch_tp / (switch_tp + switch_fn) if (switch_tp + switch_fn) > 0 else 0
    switch_f1 = 2 * switch_precision * switch_recall / (switch_precision + switch_recall) if (
                                                                                                         switch_precision + switch_recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        # Per-class metrics
        'class_0_f1': f1[0],
        'class_1_f1': f1[1],
        'class_2_f1': f1[2],
        'class_3_f1': f1[3],
        # Switch detection
        'switch_precision': switch_precision,
        'switch_recall': switch_recall,
        'switch_f1': switch_f1,
    }


def print_detailed_evaluation(model, tokenizer, test_csv, device):
    """Print detailed evaluation with per-class breakdown."""
    print("\n" + "=" * 80)
    print("DETAILED TEST SET EVALUATION")
    print("=" * 80)

    test_df = pd.read_csv(test_csv)
    model.eval()

    all_true_labels = []
    all_pred_labels = []

    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()
        true_labels = list(map(int, row['labels'].split(',')))

        # Get predictions
        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Align predictions
        word_ids = tokenized.word_ids()
        aligned_preds = []
        previous_word_idx = None

        for j, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                pred = predictions[0][j].item()
                aligned_preds.append(pred)
            previous_word_idx = word_idx

        # Ensure same length
        min_len = min(len(aligned_preds), len(true_labels))
        all_true_labels.extend(true_labels[:min_len])
        all_pred_labels.extend(aligned_preds[:min_len])

    # Print classification report
    label_names = ['NonSwitch-Auto', 'NonSwitch-Allo', 'Switch→Auto', 'Switch→Allo']
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_pred_labels,
                                target_names=label_names, digits=3))

    # Print confusion matrix style breakdown
    from collections import Counter
    print("\nLabel Distribution:")
    print("True labels:", Counter(all_true_labels))
    print("Predicted labels:", Counter(all_pred_labels))


def show_predictions(model, tokenizer, test_csv, num_examples=3, device='cuda'):
    """Show some prediction examples."""
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)

    test_df = pd.read_csv(test_csv)
    model.eval()

    label_names = {
        0: 'NonSwitch-Auto',
        1: 'NonSwitch-Allo',
        2: 'SWITCH→Auto',
        3: 'SWITCH→Allo'
    }

    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    for i, idx in enumerate(sample_indices):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()[:50]  # Show first 50 tokens
        true_labels = list(map(int, row['labels'].split(',')))[:50]

        print(f"\n--- Example {i + 1} (Segment: {row['segment_id']}) ---")

        # Get predictions
        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Align predictions
        word_ids = tokenized.word_ids()
        aligned_preds = []
        previous_word_idx = None

        for j, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                pred = predictions[0][j].item()
                aligned_preds.append(pred)
            previous_word_idx = word_idx

        # Show tokens with predictions
        min_len = min(len(tokens), len(aligned_preds), len(true_labels))

        for j in range(min_len):
            marker = "✓" if aligned_preds[j] == true_labels[j] else "✗"
            print(
                f"  {marker} [{j:3d}] {tokens[j]:15s} | True: {label_names[true_labels[j]]:17s} | Pred: {label_names[aligned_preds[j]]:17s}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_simple_classifier(
        train_csv='dataset/preprocessed_augmented/train_segments_with_more_auto_allo.csv',
        val_csv='dataset/preprocessed_augmented/val_segments_more_auto_allo.csv',
        test_csv='dataset/preprocessed_augmented/test_segments_original.csv',
        model_name='bert-base-multilingual-cased',
        output_dir='./simple_tibetan_cs_model',
        num_epochs=10,  # Was 15
        batch_size=8,  # Was 4
        learning_rate=2e-5,  # Was 1e-5
):
    """
    Train a simple 4-class token classifier.

    Args:
        train_csv: Path to training data CSV
        val_csv: Path to validation data CSV
        test_csv: Path to test data CSV
        model_name: HuggingFace model name
        output_dir: Where to save the model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """

    print("=" * 80)
    print("SIMPLE 4-CLASS TIBETAN CODE-SWITCHING CLASSIFIER")
    print("=" * 80)

    # Check if data files exist
    for csv_file in [train_csv, val_csv, test_csv]:
        if not os.path.exists(csv_file):
            print(f"ERROR: {csv_file} not found!")
            print("Please run the data processing step first or provide correct paths.")
            return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label={0: 'non_switch_auto', 1: 'non_switch_allo', 2: 'switch_to_auto', 3: 'switch_to_allo'},
        label2id={'non_switch_auto': 0, 'non_switch_allo': 1, 'switch_to_auto': 2, 'switch_to_allo': 3}
    )
    model = model.to(device)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SimpleCodeSwitchingDataset(train_csv, tokenizer)
    val_dataset = SimpleCodeSwitchingDataset(val_csv, tokenizer)
    test_dataset = SimpleCodeSwitchingDataset(test_csv, tokenizer)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",  # ✅
        eval_steps=30,  # ✅
        save_strategy="steps",  # ✅
        save_steps=60,  # ✅
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,  # ✅
        logging_steps=20,  # ✅
        load_best_model_at_end=True,
        metric_for_best_model='switch_f1',  # ✅
        greater_is_better=True,
        warmup_steps=100,  # ✅
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        gradient_accumulation_steps=1,  # ✅
        label_smoothing_factor=0.0,  # ✅
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # ✅ Use processing_class (not tokenizer)
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}/final_model")
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)

    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print("\nTest Results:")
    print(f"  Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"  Macro F1: {test_results['eval_macro_f1']:.3f}")
    print(f"  Macro Precision: {test_results['eval_macro_precision']:.3f}")
    print(f"  Macro Recall: {test_results['eval_macro_recall']:.3f}")
    print(f"\nPer-Class F1:")
    print(f"  Class 0 (NonSwitch-Auto): {test_results['eval_class_0_f1']:.3f}")
    print(f"  Class 1 (NonSwitch-Allo): {test_results['eval_class_1_f1']:.3f}")
    print(f"  Class 2 (Switch→Auto): {test_results['eval_class_2_f1']:.3f}")
    print(f"  Class 3 (Switch→Allo): {test_results['eval_class_3_f1']:.3f}")
    print(f"\nSwitch Detection (classes 2+3):")
    print(f"  Precision: {test_results['eval_switch_precision']:.3f}")
    print(f"  Recall: {test_results['eval_switch_recall']:.3f}")
    print(f"  F1: {test_results['eval_switch_f1']:.3f}")

    # Detailed evaluation
    print_detailed_evaluation(model, tokenizer, test_csv, device)

    # Show some examples
    show_predictions(model, tokenizer, test_csv, num_examples=3, device=device)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved to: {output_dir}/final_model")

    return trainer, model, tokenizer, test_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Train with default settings
    # trainer, model, tokenizer, results = train_simple_classifier(
    #     train_csv='train_segments_clean.csv',
    #     val_csv='val_segments.csv',
    #     test_csv='test_segments.csv',
    #     # model_name='bert-base-multilingual-cased',  # or 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
    #     model_name='OMRIDRORI/mbert-tibetan-continual-wylie-final',  # or 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
    #     output_dir='./alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_simple_loss_4_class_NER_11_10',
    #     # output_dir='./alloauto-segmentation-training/benchmark_models/simple_mBert_vanilla_benchmark_4_class_NER',
    #     num_epochs=15,
    #     batch_size=4,
    #     learning_rate=1e-5,
    #
    # )
    trainer, model, tokenizer, results = train_simple_classifier(
        train_csv='dataset/preprocessed_augmented/train_segments_with_more_auto_allo.csv',
        val_csv='dataset/preprocessed_augmented/val_segments_more_auto_allo.csv',
        test_csv='dataset/preprocessed_augmented/test_segments_original.csv',
        # model_name='bert-base-multilingual-cased',  # or 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
        model_name='bert-base-multilingual-cased',
        # or 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
        output_dir='./alloauto-segmentation-training/benchmark_models_standard/simple_mBert_vanilla_benchmark_4_class_NER_23_10_no_early',
        # output_dir='./alloauto-segmentation-training/benchmark_models/simple_mBert_vanilla_benchmark_4_class_NER',
        num_epochs=10,
        batch_size=8,
        learning_rate=2e-5,

    )


    # Optionally, you can also try the Tibetan-specific mBERT:
    # trainer, model, tokenizer, results = train_simple_classifier(
    #     model_name='OMRIDRORI/mbert-tibetan-continual-wylie-final',
    #     output_dir='./simple_tibetan_cs_model_wylie',
    # )