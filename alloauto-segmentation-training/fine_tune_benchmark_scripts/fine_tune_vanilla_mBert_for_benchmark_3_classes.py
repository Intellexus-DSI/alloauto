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

"""
Simple mBERT 3-Class baseline - merges non-switch classes
Uses EXACT SAME data splits as ALTO model

3-CLASS MAPPING:
  Class 0: No Switch (merging old Auto/Allo non-switch)
  Class 1: Switch to Auto
  Class 2: Switch to Allo
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
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# ============================================================================
# LABEL REMAPPING: 4-class → 3-class
# ============================================================================

def remap_labels_4to3(labels_4class):
    """
    Remap 4-class labels to 3-class during data loading.

    INPUT (4-class):
        0: Non-switch Auto
        1: Non-switch Allo
        2: Switch to Auto
        3: Switch to Allo

    OUTPUT (3-class):
        0: No Switch (merged 0+1)
        1: Switch to Auto (was 2)
        2: Switch to Allo (was 3)
    """
    remapped = []
    for label in labels_4class:
        if label == -100:  # Keep padding
            remapped.append(-100)
        elif label in [0, 1]:  # Both non-switches → class 0
            remapped.append(0)
        elif label == 2:  # Switch to Auto
            remapped.append(1)
        elif label == 3:  # Switch to Allo
            remapped.append(2)
        else:
            remapped.append(-100)  # Unknown
    return remapped


# ============================================================================
# DATASET CLASS (with automatic remapping)
# ============================================================================

class CodeSwitchingDataset3Class(Dataset):
    """
    Loads 4-class data and automatically remaps to 3-class.
    """

    def __init__(self, csv_file, tokenizer, max_length=512):
        print(f"Loading data from: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Verify data and show remapping stats
        print(f"  Segments: {len(self.data)}")
        print(f"  Files: {self.data['source_file'].nunique()}")
        self._print_label_stats()

    def _print_label_stats(self):
        """Show distribution before and after remapping"""
        all_4class = []
        all_3class = []

        for idx in range(len(self.data)):
            labels_4 = list(map(int, self.data.iloc[idx]['labels'].split(',')))
            labels_3 = remap_labels_4to3(labels_4)

            all_4class.extend([l for l in labels_4 if l != -100])
            all_3class.extend([l for l in labels_3 if l != -100])

        from collections import Counter
        count_4 = Counter(all_4class)
        count_3 = Counter(all_3class)
        total = len(all_4class)

        print(f"\n  Original 4-class distribution:")
        print(f"    Class 0 (NS-Auto): {count_4[0]} ({100 * count_4[0] / total:.1f}%)")
        print(f"    Class 1 (NS-Allo): {count_4[1]} ({100 * count_4[1] / total:.1f}%)")
        print(f"    Class 2 (SW→Auto): {count_4[2]} ({100 * count_4[2] / total:.1f}%)")
        print(f"    Class 3 (SW→Allo): {count_4[3]} ({100 * count_4[3] / total:.1f}%)")

        print(f"\n  Remapped 3-class distribution:")
        print(f"    Class 0 (NoSwitch): {count_3[0]} ({100 * count_3[0] / total:.1f}%)")
        print(f"    Class 1 (SW→Auto): {count_3[1]} ({100 * count_3[1] / total:.1f}%)")
        print(f"    Class 2 (SW→Allo): {count_3[2]} ({100 * count_3[2] / total:.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens'].split()

        # Load 4-class labels
        labels_4class = list(map(int, row['labels'].split(',')))

        # REMAP to 3-class
        labels_3class = remap_labels_4to3(labels_4class)

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels_3class[word_idx] if word_idx < len(labels_3class) else -100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(aligned_labels)
        }


# ============================================================================
# WEIGHTED TRAINER (3-class version)
# ============================================================================

class WeightedTrainer3Class(Trainer):
    """
    Weighted cross-entropy loss for 3 classes.

    Class weights:
        0: NoSwitch = 1.0 (normal weight)
        1: Switch→Auto = switch_weight (default 30.0)
        2: Switch→Allo = switch_weight (default 30.0)

    How it works:
        - Standard cross-entropy: loss = -log(p_correct_class)
        - Weighted: loss = -log(p_correct_class) * class_weight
        - Switches get 30x higher loss → model focuses on them
        - Without weighting, switches would be ignored (too rare)
    """

    def __init__(self, switch_weight=30.0, *args, **kwargs):
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.get('tokenizer')
        super().__init__(*args, **kwargs)

        # Define class weights for 3 classes
        self.class_weights = torch.tensor([
            1.0,  # Class 0: NoSwitch (normal weight)
            switch_weight,  # Class 1: Switch→Auto (30x weight)
            switch_weight  # Class 2: Switch→Allo (30x weight)
        ])

        print(f"\n📊 Class weights initialized:")
        print(f"   Class 0 (NoSwitch): {self.class_weights[0]:.1f}")
        print(f"   Class 1 (Switch→Auto): {self.class_weights[1]:.1f}")
        print(f"   Class 2 (Switch→Allo): {self.class_weights[2]:.1f}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted cross-entropy loss.

        Steps:
            1. Get labels and remove from inputs
            2. Forward pass through model → get logits
            3. Apply weighted cross-entropy loss
            4. Return loss (and outputs if needed)
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Move weights to same device as logits
        device = logits.device
        self.class_weights = self.class_weights.to(device)

        batch_size, seq_len, num_classes = logits.shape

        # Weighted cross-entropy loss
        # - Flattens batch for loss calculation
        # - ignore_index=-100 skips padding tokens
        # - reduction='mean' averages loss across all tokens
        loss = nn.functional.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='mean',
            ignore_index=-100
        )

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# EVALUATION METRICS (same proximity-aware evaluation)
# ============================================================================

def evaluate_with_proximity_3class(true_labels, pred_labels, tolerance=5):
    """
    Proximity-aware evaluation for 3-class predictions.

    Note: true_labels are still in 4-class format from test CSV.
    We need to remap them to 3-class for fair comparison.
    """
    # Remap true labels to 3-class
    true_labels_3class = np.array(remap_labels_4to3(true_labels.tolist()))
    pred_labels = np.array(pred_labels)

    # Find switch positions (in 3-class: switches are classes 1 and 2)
    true_auto = np.where(true_labels_3class == 1)[0]  # Switch→Auto
    true_allo = np.where(true_labels_3class == 2)[0]  # Switch→Allo
    pred_auto = np.where(pred_labels == 1)[0]
    pred_allo = np.where(pred_labels == 2)[0]

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

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_switches': total_true,
        'pred_switches': total_pred,
        'matched': total_matched
    }


def compute_metrics_3class(eval_pred):
    """Compute metrics for 3-class predictions"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    all_preds = predictions.flatten()
    all_labels = labels.flatten()

    mask = all_labels != -100
    all_preds = all_preds[mask]
    all_labels = all_labels[mask]

    # Remap true labels to 3-class
    all_labels_3class = remap_labels_4to3(all_labels.tolist())

    accuracy = (all_preds == all_labels_3class).mean()
    prox = evaluate_with_proximity_3class(all_labels, all_preds, tolerance=5)

    return {
        'accuracy': float(accuracy),
        'precision': float(prox['precision']),
        'recall': float(prox['recall']),
        'f1': float(prox['f1']),
        'true_switches': int(prox['true_switches']),
        'pred_switches': int(prox['pred_switches']),
    }


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_mbert_3class_baseline():
    print("=" * 80)
    print("TRAINING mBERT 3-CLASS BASELINE")
    print("Merging Non-Switch Auto + Allo into single 'NoSwitch' class")
    print("Using EXACT SAME data splits as ALTO model")
    print("=" * 80)

    # Find data files
    if os.path.exists('train_segments_clean.csv'):
        print("\n✓ Found cleaned splits")
        TRAIN_FILE = 'train_segments_clean.csv'
        VAL_FILE = 'val_segments.csv'
        TEST_FILE = 'test_segments.csv'
    else:
        print("\n❌ ERROR: No data splits found!")
        return None, None, None

    OUTPUT_DIR = './alloauto-segmentation-training/benchmark_models/mbert_3class_baseline_model_clean_train'

    # Load model (3 labels now!)
    model_name = 'bert-base-multilingual-cased'
    print(f"\nLoading {model_name} with 3 output classes...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,  # CHANGED FROM 4 to 3
        label2id={'no_switch': 0, 'switch_to_auto': 1, 'switch_to_allo': 2},
        id2label={0: 'no_switch', 1: 'switch_to_auto', 2: 'switch_to_allo'}
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # Load datasets (with automatic remapping)
    print(f"\nLoading datasets (will remap 4-class → 3-class automatically):")
    train_dataset = CodeSwitchingDataset3Class(TRAIN_FILE, tokenizer)
    val_dataset = CodeSwitchingDataset3Class(VAL_FILE, tokenizer)
    test_dataset = CodeSwitchingDataset3Class(TEST_FILE, tokenizer)

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

    # Trainer with 3-class weighted loss
    trainer = WeightedTrainer3Class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_3class,
        switch_weight=30.0  # Same 30x weight for switches
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

    print(f"\n=== mBERT 3-Class Baseline Test Results ===")
    print(f"Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Precision: {test_results['eval_precision']:.3f}")
    print(f"Recall: {test_results['eval_recall']:.3f}")
    print(f"F1: {test_results['eval_f1']:.3f}")
    print(f"True Switches: {test_results['eval_true_switches']}")
    print(f"Pred Switches: {test_results['eval_pred_switches']}")

    return trainer, model, tokenizer


if __name__ == "__main__":
    trainer, model, tokenizer = train_mbert_3class_baseline()