"""
Simple 3-Class Token Classification for Tibetan Text
Classes:
  0: No Switch (merged Auto + Allo non-switching)
  1: Switch TO Auto
  2: Switch TO Allo

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

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change GPU if needed

print(f"Number of GPUs available: {torch.cuda.device_count()}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


# ============================================================================
# LABEL MAPPING
# ============================================================================

def remap_labels(labels):
    """
    Remap 4-class labels to 3-class labels:
    - 0 (NonSwitch-Auto) -> 0 (NoSwitch)
    - 1 (NonSwitch-Allo) -> 0 (NoSwitch)
    - 2 (Switch→Auto) -> 1
    - 3 (Switch→Allo) -> 2
    """
    remapped = []
    for label in labels:
        if label == -100:  # Keep padding labels
            remapped.append(-100)
        elif label in [0, 1]:  # Merge non-switching classes
            remapped.append(0)
        elif label == 2:  # Switch to Auto
            remapped.append(1)
        elif label == 3:  # Switch to Allo
            remapped.append(2)
        else:
            remapped.append(-100)  # Unknown labels
    return remapped


# ============================================================================
# DATASET CLASS
# ============================================================================

class Simple3ClassDataset(Dataset):
    """Simple dataset for 3-class token-level classification."""

    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loaded {len(self.data)} segments from {csv_file}")

        # Verify label distribution after remapping
        self._print_label_distribution()

    def _print_label_distribution(self):
        """Print distribution of labels after remapping."""
        all_labels = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            labels = list(map(int, row['labels'].split(',')))
            remapped = remap_labels(labels)
            all_labels.extend([l for l in remapped if l != -100])

        from collections import Counter
        label_counts = Counter(all_labels)
        total = len(all_labels)

        print(f"  Label distribution after remapping:")
        print(f"    Class 0 (NoSwitch): {label_counts[0]} ({label_counts[0] / total * 100:.1f}%)")
        print(f"    Class 1 (Switch→Auto): {label_counts[1]} ({label_counts[1] / total * 100:.1f}%)")
        print(f"    Class 2 (Switch→Allo): {label_counts[2]} ({label_counts[2] / total * 100:.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens'].split()
        labels = list(map(int, row['labels'].split(',')))

        # Remap to 3 classes
        labels = remap_labels(labels)

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
    """Compute standard classification metrics for 3 classes."""
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
        true_labels, pred_labels, average=None, labels=[0, 1, 2], zero_division=0
    )

    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Switch detection metrics (classes 1 and 2 combined)
    switch_mask_true = np.array([l in [1, 2] for l in true_labels])
    switch_mask_pred = np.array([l in [1, 2] for l in pred_labels])

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

        # Remap to 3 classes
        true_labels = remap_labels(true_labels)

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
    label_names = ['NoSwitch', 'Switch→Auto', 'Switch→Allo']
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
        0: 'NoSwitch',
        1: 'SWITCH→Auto',
        2: 'SWITCH→Allo'
    }

    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    for i, idx in enumerate(sample_indices):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()[:50]  # Show first 50 tokens
        true_labels = list(map(int, row['labels'].split(',')))[:50]

        # Remap to 3 classes
        true_labels = remap_labels(true_labels)

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
                f"  {marker} [{j:3d}] {tokens[j]:15s} | True: {label_names[true_labels[j]]:13s} | Pred: {label_names[aligned_preds[j]]:13s}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_3class_classifier(
        train_csv='dataset/preprocessed_augmented/train_segments_with_more_auto_allo.csv',
        val_csv='dataset/preprocessed_augmented/val_segments_more_auto_allo.csv',
        test_csv='dataset/preprocessed_augmented/test_segments_original.csv',
        model_name='bert-base-multilingual-cased',
        output_dir='./simple_3class_tibetan_cs_model',
        num_epochs=10,
        batch_size=8,
        learning_rate=2e-5,
):
    """
    Train a simple 3-class token classifier.

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
    print("SIMPLE 3-CLASS TIBETAN CODE-SWITCHING CLASSIFIER")
    print("Classes: 0=NoSwitch, 1=Switch→Auto, 2=Switch→Allo")
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
        num_labels=3,
        id2label={0: 'no_switch', 1: 'switch_to_auto', 2: 'switch_to_allo'},
        label2id={'no_switch': 0, 'switch_to_auto': 1, 'switch_to_allo': 2}
    )
    model = model.to(device)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = Simple3ClassDataset(train_csv, tokenizer)
    val_dataset = Simple3ClassDataset(val_csv, tokenizer)
    test_dataset = Simple3ClassDataset(test_csv, tokenizer)

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
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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
    print(f"  Class 0 (NoSwitch): {test_results['eval_class_0_f1']:.3f}")
    print(f"  Class 1 (Switch→Auto): {test_results['eval_class_1_f1']:.3f}")
    print(f"  Class 2 (Switch→Allo): {test_results['eval_class_2_f1']:.3f}")
    print(f"\nSwitch Detection (classes 1+2):")
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

def remap_3class_to_4class(predictions_3class):
    """
    Remap 3-class predictions to 4-class format.
    3-class: 0=NoSwitch, 1=Switch→Auto, 2=Switch→Allo
    4-class: 0=NonSwitch-Auto, 1=NonSwitch-Allo, 2=Switch→Auto, 3=Switch→Allo
    """
    remapped = []
    for pred in predictions_3class:
        if pred == -100:
            remapped.append(-100)
        elif pred == 0:  # NoSwitch → NonSwitch-Auto
            remapped.append(0)
        elif pred == 1:  # Switch→Auto
            remapped.append(2)
        elif pred == 2:  # Switch→Allo
            remapped.append(3)
        else:
            remapped.append(0)
    return remapped
def verify_3class_predictions(test_csv='./test_segments.csv', model_path='./path/to/simple_3class_model'):
    """
    Verify that 3-class model predictions are being remapped correctly.
    """
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    print("=" * 80)
    print("VERIFYING 3-CLASS MODEL PREDICTIONS AND REMAPPING")
    print("=" * 80)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Load test data
    test_df = pd.read_csv(test_csv)

    # Check a few segments
    for idx in range(min(20, len(test_df))):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()[:50]  # First 50 tokens
        true_labels = [int(l) for l in row['labels'].split(',')][:50]

        print(f"\n--- Segment {idx} ---")

        # Get raw 3-class predictions
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
            logits = outputs.logits
            predictions_3class = torch.argmax(logits, dim=2)

        # Align predictions
        word_ids = tokenizer_output.word_ids()
        aligned_3class = []
        previous_word_idx = None

        for j, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                aligned_3class.append(predictions_3class[0][j].item())
            previous_word_idx = word_idx

        # Remap to 4-class


        aligned_4class = remap_3class_to_4class(aligned_3class[:len(tokens)])

        # Show comparison
        print(f"\nFirst 20 tokens:")
        print(f"{'Token':<15} {'True(4cls)':<12} {'Pred(3cls)':<12} {'Remapped(4cls)':<15}")
        print("-" * 60)

        for i in range(min(20, len(tokens))):
            true_label = true_labels[i] if i < len(true_labels) else -1
            pred_3cls = aligned_3class[i] if i < len(aligned_3class) else -1
            pred_4cls = aligned_4class[i] if i < len(aligned_4class) else -1

            # Label names
            true_name = {0: 'NS-Auto', 1: 'NS-Allo', 2: 'SW→Auto', 3: 'SW→Allo'}.get(true_label, '?')
            pred_3_name = {0: 'NoSwitch', 1: 'SW→Auto', 2: 'SW→Allo'}.get(pred_3cls, '?')
            pred_4_name = {0: 'NS-Auto', 1: 'NS-Allo', 2: 'SW→Auto', 3: 'SW→Allo'}.get(pred_4cls, '?')

            marker = '✓' if pred_4cls == true_label else '✗'

            print(f"{marker} {tokens[i]:<15} {true_name:<12} {pred_3_name:<12} {pred_4_name:<15}")

        # Count switches
        true_switches = sum(1 for l in true_labels[:len(tokens)] if l in [2, 3])
        pred_3_switches = sum(1 for l in aligned_3class[:len(tokens)] if l in [1, 2])
        pred_4_switches = sum(1 for l in aligned_4class[:len(tokens)] if l in [2, 3])

        print(f"\nSwitch counts:")
        print(f"  True switches (4-class): {true_switches}")
        print(f"  Predicted switches (3-class raw): {pred_3_switches}")
        print(f"  Predicted switches (4-class remapped): {pred_4_switches}")

        if pred_3_switches != pred_4_switches:
            print(f"  ⚠️ WARNING: Switch count mismatch after remapping!")
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Train with default settings
    trainer, model, tokenizer, results = train_3class_classifier(
        train_csv='dataset/preprocessed_augmented/train_segments_with_more_auto_allo.csv',
        val_csv='dataset/preprocessed_augmented/val_segments_more_auto_allo.csv',
        test_csv='dataset/preprocessed_augmented/test_segments_original.csv',
        model_name='bert-base-multilingual-cased',  # or 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
        output_dir='./alloauto-segmentation-training/benchmark_models_standard/simple_mBert_vanilla_benchmark_3_class_NER_23_10_no_early',
        # output_dir='./alloauto-segmentation-training/benchmark_models/simple_mBert_vanilla_benchmark_3_class_NER',
        num_epochs=10,
        batch_size=8,
        learning_rate=2e-5,
    )
    # verify_3class_predictions(
    #     test_csv='./test_segments.csv',
    #     model_path='./alloauto-segmentation-training/benchmark_models/simple_mBert_vanilla_benchmark_3_class_NER/final_model'
    # )

    # Optionally, you can also try the Tibetan-specific mBERT:
    # trainer, model, tokenizer, results = train_3class_classifier(
    #     model_name='OMRIDRORI/mbert-tibetan-continual-wylie-final',
    #     output_dir='./simple_3class_tibetan_cs_model_wylie',
    # )