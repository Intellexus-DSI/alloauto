"""
4-Class ALTO with ADDITIVE Loss (Multiple Models Version) - NO SEGMENTATION
This version is identical to the original but WITHOUT segmentation rewards/penalties
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

# ============================================================================
# CONFIGURATION: MODELS TO TRAIN
# ============================================================================

# Define your models and output directories here
MODELS_TO_TRAIN = [
    {   #ALTO additive - NO SEGMENTATION
        'model_name': 'OMRIDRORI/mbert-tibetan-continual-wylie-final',
        'output_dir': './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_additive_loss_NO_SEG_REWARD_23_10'
    },
    # {   #CINO
    #     'model_name': 'hfl/cino-base-v2',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models/CINO_baseline_model_additive_loss_NO_SEG_10_10'
    #
    # },
    # {   # XLM Roberta
    #     'model_name': 'xlm-roberta-base',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models/XLM_roberta_baseline_model_additive_loss_NO_SEG_10_10'
    # },
    # {   # mBERT
    #     'model_name': 'bert-base-multilingual-cased',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models/mBERT_baseline_model_additive_loss_NO_SEG_10_10'
    # },
    # {   # tibetian roberta
    #     'model_name': 'sangjeedondrub/tibetan-roberta-base',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models/tibetan_roberta_baseline_model_additive_loss_NO_SEG_10_10'
    # },
]

# Dataset files New Augmented 23/10
TRAIN_FILE = 'dataset/preprocessed_augmented/train_segments_with_more_auto_allo.csv'
VAL_FILE = 'dataset/preprocessed_augmented/val_segments_more_auto_allo.csv'
TEST_FILE = 'dataset/preprocessed_augmented/test_segments_original.csv'

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# ============================================================================
# DATASET CLASS
# ============================================================================

class CodeSwitchingDataset4Class(Dataset):
    """Dataset for 4-class token-level code-switching."""

    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"  Loading {csv_file}: {len(self.data)} segments")

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
# ADDITIVE LOSS - NO SEGMENTATION VERSION
# ============================================================================

class SwitchFocusedLossAdditiveNoSegmentation(nn.Module):
    """ADDITIVE version of SwitchFocusedLoss WITHOUT segmentation rewards/penalties"""

    def __init__(self,
                 switch_recall_weight=10.0,
                 proximity_tolerance=5,
                 proximity_reward=2.0,
                 far_penalty=1.5):
        super().__init__()
        self.proximity_tolerance = proximity_tolerance
        self.proximity_reward = proximity_reward
        self.far_penalty = far_penalty

        self.class_weights = torch.tensor([
            0.1,  # Nearly ignore non-switch classes
            0.1,
            5.0,  # Focus on switches
            5.0
        ])

    def forward(self, logits, labels, seg_marks=None):
        """
        Forward pass WITHOUT segmentation alignment checks
        seg_marks parameter kept for compatibility but not used
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

        # ADDITIVE adjustments (NO SEGMENTATION)
        additive_adjustments = torch.zeros_like(base_loss)

        # Proximity logic with ADDITIVE adjustments (ONLY proximity, no segmentation)
        for b in range(batch_size):
            true_switches = torch.where(labels[b] >= 2)[0]
            pred_switches = torch.where(predictions[b] >= 2)[0]

            # REWARD predictions near true switches
            for true_pos in true_switches:
                window_start = max(0, true_pos - self.proximity_tolerance)
                window_end = min(seq_len, true_pos + self.proximity_tolerance + 1)
                window_preds = predictions[b, window_start:window_end]

                if torch.any(window_preds >= 2):
                    additive_adjustments[b, true_pos] -= self.proximity_reward

            # Mild penalty for predictions far from any true switch
            for pred_pos in pred_switches:
                if len(true_switches) > 0:
                    distances = torch.abs(true_switches - pred_pos)
                    min_distance = torch.min(distances).item()

                    if min_distance > self.proximity_tolerance:
                        additive_adjustments[b, pred_pos] += self.far_penalty

        # Combine base loss with additive adjustments
        adjusted_loss = base_loss + additive_adjustments

        # Only consider non-padding positions
        mask = (labels != -100).float()
        adjusted_loss = (adjusted_loss * mask).sum() / mask.sum()

        return adjusted_loss


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics_for_trainer(eval_pred, tolerance=5):
    """
    Compute metrics with proximity tolerance for switch detection.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Flatten and filter out padding (-100)
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

    # Switch detection metrics (classes 2 and 3)
    true_switches = np.where(flat_labels >= 2)[0]
    pred_switches = np.where(flat_preds >= 2)[0]

    # Proximity-based true positives
    true_positives = 0
    for true_pos in true_switches:
        window_start = max(0, true_pos - tolerance)
        window_end = min(len(flat_preds), true_pos + tolerance + 1)
        if np.any(flat_preds[window_start:window_end] >= 2):
            true_positives += 1

    # Calculate precision, recall, F1
    recall = true_positives / len(true_switches) if len(true_switches) > 0 else 0.0
    precision = true_positives / len(pred_switches) if len(pred_switches) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-type metrics
    to_auto_mask = flat_labels == 2
    to_allo_mask = flat_labels == 3

    to_auto_tp = np.sum((flat_preds == 2) & to_auto_mask)
    to_auto_fp = np.sum((flat_preds == 2) & ~to_auto_mask)
    to_auto_fn = np.sum((flat_preds != 2) & to_auto_mask)

    to_allo_tp = np.sum((flat_preds == 3) & to_allo_mask)
    to_allo_fp = np.sum((flat_preds == 3) & ~to_allo_mask)
    to_allo_fn = np.sum((flat_preds != 3) & to_allo_mask)

    to_auto_precision = to_auto_tp / (to_auto_tp + to_auto_fp) if (to_auto_tp + to_auto_fp) > 0 else 0.0
    to_auto_recall = to_auto_tp / (to_auto_tp + to_auto_fn) if (to_auto_tp + to_auto_fn) > 0 else 0.0

    to_allo_precision = to_allo_tp / (to_allo_tp + to_allo_fp) if (to_allo_tp + to_allo_fp) > 0 else 0.0
    to_allo_recall = to_allo_tp / (to_allo_tp + to_allo_fn) if (to_allo_tp + to_allo_fn) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'switch_precision': precision,
        'switch_recall': recall,
        'switch_f1': f1,
        'true_switches': len(true_switches),
        'pred_switches': len(pred_switches),
        'to_auto_precision': to_auto_precision,
        'to_auto_recall': to_auto_recall,
        'to_allo_precision': to_allo_precision,
        'to_allo_recall': to_allo_recall,
    }


# ============================================================================
# TRANSITION CONSTRAINTS
# ============================================================================

def apply_transition_constraints(predictions):
    """Apply linguistic transition constraints to predictions."""
    predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    predictions = predictions.copy()

    for i in range(1, len(predictions)):
        prev = predictions[i - 1]
        curr = predictions[i]

        # Invalid transitions
        if prev == 0 and curr == 3:  # auto → to_allo (should be auto → to_auto)
            predictions[i] = 2
        elif prev == 1 and curr == 2:  # allo → to_auto (should be allo → to_allo)
            predictions[i] = 3
        elif prev == 2 and curr == 2:  # to_auto → to_auto (should be non_switch_auto)
            predictions[i] = 0
        elif prev == 3 and curr == 3:  # to_allo → to_allo (should be non_switch_allo)
            predictions[i] = 1
        elif prev == 2 and curr == 1:  # to_auto → non_switch_allo
            predictions[i] = 0
        elif prev == 3 and curr == 0:  # to_allo → non_switch_auto
            predictions[i] = 1

    return predictions


# ============================================================================
# CUSTOM TRAINER - NO SEGMENTATION VERSION
# ============================================================================

class SimpleSwitchTrainerAdditiveNoSegmentation(Trainer):
    """
    Custom Trainer with ADDITIVE loss and NO segmentation penalties/rewards
    """

    def __init__(self,
                 switch_recall_weight=10.0,
                 proximity_tolerance=5,
                 proximity_reward=2.0,
                 far_penalty=1.5,
                 apply_constraints=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize loss function WITHOUT segmentation parameters
        self.custom_loss = SwitchFocusedLossAdditiveNoSegmentation(
            switch_recall_weight=switch_recall_weight,
            proximity_tolerance=proximity_tolerance,
            proximity_reward=proximity_reward,
            far_penalty=far_penalty
        )

        self.apply_constraints = apply_constraints
        print("\n✓ Trainer initialized with ADDITIVE loss (NO SEGMENTATION)")
        print(f"  Proximity tolerance: {proximity_tolerance}")
        print(f"  Proximity reward: {proximity_reward}")
        print(f"  Far penalty: {far_penalty}")
        print(f"  Apply constraints: {apply_constraints}")
        print("  NOTE: Segmentation penalties/rewards DISABLED\n")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation WITHOUT segmentation marks
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Call custom loss WITHOUT seg_marks parameter
        loss = self.custom_loss(logits, labels, seg_marks=None)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Prediction step with optional transition constraints
        """
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            loss = self.custom_loss(logits, labels, seg_marks=None)

        predictions = torch.argmax(logits, dim=-1)

        if self.apply_constraints:
            batch_size = predictions.shape[0]
            for b in range(batch_size):
                predictions[b] = torch.tensor(
                    apply_transition_constraints(predictions[b]),
                    device=predictions.device
                )

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Evaluation with transition constraints applied
        """
        outputs = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.apply_constraints:
            logits = outputs.predictions if hasattr(outputs, 'predictions') else None
            labels = outputs.label_ids if hasattr(outputs, 'label_ids') else None

            if logits is not None and labels is not None:
                predictions = np.argmax(logits, axis=-1)
                batch_size = predictions.shape[0]

                for b in range(batch_size):
                    predictions[b] = apply_transition_constraints(predictions[b])

                # Recompute metrics with constrained predictions
                constrained_metrics = self.compute_metrics((predictions, labels))
                for key, value in constrained_metrics.items():
                    outputs[f"{metric_key_prefix}_{key}"] = value

        return outputs

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """
        Prediction with optional transition constraints
        """
        outputs = super().predict(test_dataset, ignore_keys, metric_key_prefix)

        if self.apply_constraints:
            predictions = outputs.predictions
            labels = outputs.label_ids
            batch_size = predictions.shape[0]

            for b in range(batch_size):
                predictions[b] = torch.tensor(
                    apply_transition_constraints(predictions[b]),
                    device=predictions.device
                )

            return (loss, logits, labels)

        return outputs


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_alto_4class_additive_no_segmentation(model_name, output_dir):
    """
    Main training pipeline - ADDITIVE version WITHOUT segmentation

    Args:
        model_name: Hugging Face model name or path
        output_dir: Directory to save the fine-tuned model
    """
    print("=" * 80)
    print("ALTO 4-CLASS with ADDITIVE LOSS (NO SEGMENTATION)")
    print("=" * 80)
    print(f"Base Model: {model_name}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # Check if files exist
    for f in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found!")
            return None

    print(f"\n✓ Using datasets:")
    print(f"  Train: {TRAIN_FILE}")
    print(f"  Val: {VAL_FILE}")
    print(f"  Test: {TEST_FILE}")

    # Initialize model
    print("\n" + "=" * 80)
    print("STEP 1: Initializing 4-class model")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if model_name == 'sangjeedondrub/tibetan-roberta-base':
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            add_prefix_space=True  # <-- REQUIRED for RoBERTa when is_split_into_words=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=4,
        label2id={'non_switch_auto': 0, 'non_switch_allo': 1, 'to_auto': 2, 'to_allo': 3},
        id2label={0: 'non_switch_auto', 1: 'non_switch_allo', 2: 'to_auto', 3: 'to_allo'}
    )

    with torch.no_grad():
        model.classifier.bias.data[0] = 0.0
        model.classifier.bias.data[1] = 0.0
        model.classifier.bias.data[2] = -1.0
        model.classifier.bias.data[3] = -1.0

    model = model.to(device)
    print(f"✓ Model initialized with 4 classes")

    # Create datasets
    print("\n" + "=" * 80)
    print("STEP 2: Loading datasets")
    print("=" * 80)

    train_dataset = CodeSwitchingDataset4Class(TRAIN_FILE, tokenizer)
    val_dataset = CodeSwitchingDataset4Class(VAL_FILE, tokenizer)
    test_dataset = CodeSwitchingDataset4Class(TEST_FILE, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    print(f"✓ Datasets loaded")

    # Training setup
    print("\n" + "=" * 80)
    print("STEP 3: Setting up training with ADDITIVE loss (NO SEGMENTATION)")
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

    trainer = SimpleSwitchTrainerAdditiveNoSegmentation(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_for_trainer(eval_pred, tolerance=5),
        switch_recall_weight=10.0,
        proximity_tolerance=5,
        proximity_reward=2.0,
        far_penalty=1.5
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    # Train
    print("\n" + "=" * 80)
    print("STEP 4: Training")
    print("=" * 80)
    trainer.train()

    # Save
    final_model_path = f'{output_dir}/final_model'
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✓ Model saved to: {final_model_path}")

    # Evaluate
    print("\n" + "=" * 80)
    print("STEP 5: Evaluating on test set")
    print("=" * 80)
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\n{'=' * 80}")
    print(f"FINAL TEST RESULTS")
    print(f"{'=' * 80}")
    print(f"Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Switch F1: {test_results['eval_switch_f1']:.3f}")
    print(f"Switch Precision: {test_results['eval_switch_precision']:.3f}")
    print(f"Switch Recall: {test_results['eval_switch_recall']:.3f}")
    print(f"True Switches: {test_results['eval_true_switches']}")
    print(f"Predicted Switches: {test_results['eval_pred_switches']}")
    print(f"\nPer-Type Performance:")
    print(f"  Switch→Auto Precision: {test_results.get('eval_to_auto_precision', 0):.3f}")
    print(f"  Switch→Auto Recall: {test_results.get('eval_to_auto_recall', 0):.3f}")
    print(f"  Switch→Allo Precision: {test_results.get('eval_to_allo_precision', 0):.3f}")
    print(f"  Switch→Allo Recall: {test_results.get('eval_to_allo_recall', 0):.3f}")

    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'=' * 80}")

    return trainer, model, tokenizer, test_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING MULTI-MODEL TRAINING (NO SEGMENTATION)")
    print("=" * 80)
    print(f"\nTotal models to train: {len(MODELS_TO_TRAIN)}")

    all_results = []

    for idx, model_config in enumerate(MODELS_TO_TRAIN, 1):
        model_name = model_config['model_name']
        output_dir = model_config['output_dir']

        print("\n" + "=" * 80)
        print(f"TRAINING MODEL {idx}/{len(MODELS_TO_TRAIN)}")
        print("=" * 80)

        result = train_alto_4class_additive_no_segmentation(
            model_name=model_name,
            output_dir=output_dir
        )

        if result is not None:
            trainer, model, tokenizer, test_results = result
            all_results.append({
                'model_name': model_name,
                'output_dir': output_dir,
                'results': test_results
            })
            print(f"\n✓ Successfully trained: {model_name}")
        else:
            print(f"\n✗ Failed to train: {model_name}")

        print("\n")

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for result in all_results:
        print(f"\nModel: {result['model_name']}")
        print(f"  Switch F1: {result['results']['eval_switch_f1']:.3f}")
        print(f"  Accuracy: {result['results']['eval_accuracy']:.3f}")
        print(f"  Output: {result['output_dir']}")

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE!")
    print("=" * 80)