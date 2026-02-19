"""
3-Class ALTO with ADDITIVE Loss - NO SEGMENTATION PENALTY (Multiple Models Version)
====================================================================================
Converts 4-class to 3-class with IDENTICAL additive loss logic:
- Class 0: Non-switch (was non_switch_auto + non_switch_allo)
- Class 1: Switch→Auto
- Class 2: Switch→Allo

EXACT SAME parameters as 4-class version (EXCEPT SEGMENTATION):
- switch_recall_weight: 10.0
- proximity_tolerance: 5
- segmentation_penalty: DISABLED (no reward/punishment for "/" or "//")
- segmentation_reward: DISABLED (no reward/punishment for "/" or "//")
- proximity_reward: 2.0
- far_penalty: 1.5
- class_weights: [0.1, 5.0, 5.0] (adjusted for 3 classes)
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

MODELS_TO_TRAIN = [
    {  # ALTO additive 3-class - NO SEGMENTATION
        'model_name': 'OMRIDRORI/mbert-tibetan-continual-wylie-final',
        'output_dir': './alloauto-segmentation-training/benchmark_models_ALTO_architecture/ALTO_additive_NO_SEG_3class_24_10'
    },
    # {   # CINO
    #     'model_name': 'hfl/cino-base-v2',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models_ALTO_architecture/CINO_additive_NO_SEG_3class_24_10'
    # },
    # {   # XLM Roberta
    #     'model_name': 'xlm-roberta-base',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models_ALTO_architecture/XLM_roberta_additive_NO_SEG_3class_24_10'
    # },
    # {   # mBERT
    #     'model_name': 'bert-base-multilingual-cased',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mBERT_additive_NO_SEG_3class_23_10'
    # },
    # {   # tibetian roberta
    #     'model_name': 'sangjeedondrub/tibetan-roberta-base',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models_ALTO_architecture/tibetan_roberta_additive_NO_SEG_3class_24_10'
    # },
]

# Dataset files
TRAIN_FILE = 'dataset/preprocessed_augmented/train_segments_with_more_auto_allo.csv'
VAL_FILE = 'dataset/preprocessed_augmented/val_segments_more_auto_allo.csv'
TEST_FILE = 'dataset/preprocessed_augmented/test_segments_original.csv'

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# ============================================================================
# LABEL REMAPPING
# ============================================================================

def remap_4class_to_3class(label_4class):
    """Remap 4-class labels to 3-class"""
    if label_4class == -100:
        return -100
    elif label_4class in [0, 1]:  # non_switch_auto or non_switch_allo → non_switch
        return 0
    elif label_4class == 2:  # switch→auto
        return 1
    elif label_4class == 3:  # switch→allo
        return 2
    else:
        return 0


def verify_remapping_preserves_switches(csv_file):
    """Verify that 4→3 class remapping doesn't change switch counts"""
    df = pd.read_csv(csv_file)

    mismatches = []
    for idx in range(len(df)):
        labels_4class = [int(l) for l in df.iloc[idx]['labels'].split(',')]
        labels_3class = [remap_4class_to_3class(l) for l in labels_4class]

        # Count switches
        switches_4class = sum(1 for l in labels_4class if l in [2, 3])
        switches_3class = sum(1 for l in labels_3class if l in [1, 2])

        # Count by type
        auto_4class = sum(1 for l in labels_4class if l == 2)
        auto_3class = sum(1 for l in labels_3class if l == 1)
        allo_4class = sum(1 for l in labels_4class if l == 3)
        allo_3class = sum(1 for l in labels_3class if l == 2)

        if switches_4class != switches_3class or auto_4class != auto_3class or allo_4class != allo_3class:
            mismatches.append({
                'segment': idx,
                'switches_4': switches_4class,
                'switches_3': switches_3class,
                'auto_4': auto_4class,
                'auto_3': auto_3class,
                'allo_4': allo_4class,
                'allo_3': allo_3class
            })

    if len(mismatches) == 0:
        print(f"✅ VERIFIED: All switches preserved in {csv_file}")
        return True
    else:
        print(f"❌ WARNING: {len(mismatches)} segments have switch count mismatches!")
        for m in mismatches[:5]:  # Show first 5
            print(f"  Segment {m['segment']}: "
                  f"Total: {m['switches_4']}→{m['switches_3']}, "
                  f"Auto: {m['auto_4']}→{m['auto_3']}, "
                  f"Allo: {m['allo_4']}→{m['allo_3']}")
        return False


# Run this on all datasets
print("Verifying train data...")
verify_remapping_preserves_switches(TRAIN_FILE)
print("\nVerifying val data...")
verify_remapping_preserves_switches(VAL_FILE)
print("\nVerifying test data...")
verify_remapping_preserves_switches(TEST_FILE)


# ============================================================================
# 3-CLASS DATASET
# ============================================================================

class CodeSwitchingDataset3Class(Dataset):
    """Dataset for 3-class token-level code-switching with automatic remapping."""

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
        labels_4class = list(map(int, row['labels'].split(',')))

        # Remap to 3-class
        labels_3class = [remap_4class_to_3class(l) for l in labels_4class]

        encoding = self.tokenize_and_align_labels(tokens, labels_3class)

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
# 3-CLASS ADDITIVE LOSS
# ============================================================================

class SwitchFocusedLossAdditive3Class(nn.Module):
    """3-CLASS ADDITIVE version of SwitchFocusedLoss - EXACT SAME logic"""

    def __init__(self,
                 switch_recall_weight=10.0,
                 proximity_tolerance=5,
                 segmentation_penalty=2.0,
                 segmentation_reward=1.0,
                 proximity_reward=2.0,
                 far_penalty=1.5):
        super().__init__()
        self.proximity_tolerance = proximity_tolerance
        self.segmentation_penalty = segmentation_penalty
        self.segmentation_reward = segmentation_reward
        self.proximity_reward = proximity_reward
        self.far_penalty = far_penalty

        # 3-class weights (same pattern as 4-class)
        self.class_weights = torch.tensor([
            0.1,  # Nearly ignore non-switch class
            5.0,  # Focus on switch→auto
            5.0  # Focus on switch→allo
        ])

    def check_segmentation_alignment(self, seg_marks, predictions, b, seq_len):
        """Check if switches align with segmentation marks - DISABLED"""
        # DISABLED: No reward/punishment for "/" or "//" delimiters
        # Return zero adjustments (no segmentation penalty/reward applied)
        return torch.zeros(seq_len, device=predictions.device)

        # ORIGINAL CODE (DISABLED):
        # seg_adjustments = torch.zeros(seq_len, device=predictions.device)
        #
        # for t in range(seq_len):
        #     # 3-class: switches are classes 1 and 2 (instead of 2 and 3)
        #     if predictions[b, t] >= 1:  # This is a predicted switch
        #
        #         # Check if switch happens 1-2 positions AFTER segmentation mark (BAD)
        #         for offset in [1, 2]:
        #             if t - offset >= 0 and seg_marks[b, t - offset] > 0:
        #                 seg_adjustments[t] += self.segmentation_penalty
        #                 break
        #
        #         # Check if switch happens AT segmentation mark (GOOD)
        #         if seg_marks[b, t] > 0:
        #             seg_adjustments[t] -= self.segmentation_reward
        #
        #         # Check if switch happens RIGHT BEFORE segmentation mark (GOOD)
        #         elif t + 1 < seq_len and seg_marks[b, t + 1] > 0:
        #             seg_adjustments[t] -= self.segmentation_reward
        #
        # return seg_adjustments

    def forward(self, logits, labels, seg_marks=None):
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device
        self.class_weights = self.class_weights.to(device)

        # Base loss with heavy switch weighting - EXACT SAME
        base_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='none'
        ).view(batch_size, seq_len)

        predictions = torch.argmax(logits, dim=-1)

        # ADDITIVE adjustments - EXACT SAME logic
        additive_adjustments = torch.zeros_like(base_loss)

        # Apply segmentation alignment if provided
        if seg_marks is not None:
            for b in range(batch_size):
                seg_adjustment = self.check_segmentation_alignment(seg_marks, predictions, b, seq_len)
                additive_adjustments[b] += seg_adjustment

        # Proximity logic with ADDITIVE adjustments - EXACT SAME
        for b in range(batch_size):
            # 3-class: switches are classes 1 and 2 (instead of 2 and 3)
            true_switches = torch.where(labels[b] >= 1)[0]
            pred_switches = torch.where(predictions[b] >= 1)[0]

            # REWARD predictions near true switches
            for true_pos in true_switches:
                window_start = max(0, true_pos - self.proximity_tolerance)
                window_end = min(seq_len, true_pos + self.proximity_tolerance + 1)
                window_preds = predictions[b, window_start:window_end]

                if torch.any(window_preds >= 1):
                    additive_adjustments[b, true_pos] -= self.proximity_reward

            # Mild penalty for predictions far from any true switch
            for pred_pos in pred_switches:
                if len(true_switches) > 0:
                    distances = torch.abs(true_switches - pred_pos)
                    min_distance = torch.min(distances).item()

                    if min_distance > self.proximity_tolerance:
                        additive_adjustments[b, pred_pos] += self.far_penalty

        # Apply ADDITIVE adjustments to base loss - EXACT SAME
        adjusted_loss = base_loss + additive_adjustments

        valid_mask = (labels != -100).float()
        return (adjusted_loss * valid_mask).sum() / valid_mask.sum()


# ============================================================================
# 3-CLASS EVALUATION FUNCTIONS
# ============================================================================

def evaluate_switch_detection_with_proximity_3class(true_labels, pred_labels, tolerance=5):
    """Evaluate 3-class switch detection with proximity tolerance and TYPE matching"""
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # 3-class: switches are classes 1 and 2
    true_switches_to_auto = np.where(true_labels == 1)[0]
    true_switches_to_allo = np.where(true_labels == 2)[0]
    pred_switches_to_auto = np.where(pred_labels == 1)[0]
    pred_switches_to_allo = np.where(pred_labels == 2)[0]

    matched_true_to_auto = set()
    matched_pred_to_auto = set()
    matched_true_to_allo = set()
    matched_pred_to_allo = set()

    exact_matches = 0
    proximity_matches = 0

    # Match Switch→Auto - EXACT SAME logic
    for pred_pos in pred_switches_to_auto:
        if len(true_switches_to_auto) > 0:
            distances = np.abs(true_switches_to_auto - pred_pos)
            min_distance = np.min(distances)
            closest_true_idx = np.argmin(distances)
            closest_true_pos = true_switches_to_auto[closest_true_idx]

            if closest_true_pos not in matched_true_to_auto:
                if min_distance == 0:
                    exact_matches += 1
                    matched_true_to_auto.add(closest_true_pos)
                    matched_pred_to_auto.add(pred_pos)
                elif min_distance <= tolerance:
                    proximity_matches += 1
                    matched_true_to_auto.add(closest_true_pos)
                    matched_pred_to_auto.add(pred_pos)

    # Match Switch→Allo - EXACT SAME logic
    for pred_pos in pred_switches_to_allo:
        if len(true_switches_to_allo) > 0:
            distances = np.abs(true_switches_to_allo - pred_pos)
            min_distance = np.min(distances)
            closest_true_idx = np.argmin(distances)
            closest_true_pos = true_switches_to_allo[closest_true_idx]

            if closest_true_pos not in matched_true_to_allo:
                if min_distance == 0:
                    exact_matches += 1
                    matched_true_to_allo.add(closest_true_pos)
                    matched_pred_to_allo.add(pred_pos)
                elif min_distance <= tolerance:
                    proximity_matches += 1
                    matched_true_to_allo.add(closest_true_pos)
                    matched_pred_to_allo.add(pred_pos)

    # Calculate metrics - EXACT SAME
    total_true_switches = len(true_switches_to_auto) + len(true_switches_to_allo)
    total_pred_switches = len(pred_switches_to_auto) + len(pred_switches_to_allo)
    total_matches = exact_matches + proximity_matches

    precision = total_matches / total_pred_switches if total_pred_switches > 0 else 0
    recall = total_matches / total_true_switches if total_true_switches > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    to_auto_precision = len(matched_pred_to_auto) / len(pred_switches_to_auto) if len(pred_switches_to_auto) > 0 else 0
    to_auto_recall = len(matched_true_to_auto) / len(true_switches_to_auto) if len(true_switches_to_auto) > 0 else 0
    to_allo_precision = len(matched_pred_to_allo) / len(pred_switches_to_allo) if len(pred_switches_to_allo) > 0 else 0
    to_allo_recall = len(matched_true_to_allo) / len(true_switches_to_allo) if len(true_switches_to_allo) > 0 else 0

    return {
        'exact_matches': exact_matches,
        'proximity_matches': proximity_matches,
        'total_matches': total_matches,
        'true_switches': total_true_switches,
        'pred_switches': total_pred_switches,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'to_auto_precision': to_auto_precision,
        'to_auto_recall': to_auto_recall,
        'to_allo_precision': to_allo_precision,
        'to_allo_recall': to_allo_recall,
        'matched_to_auto': len(matched_true_to_auto),
        'matched_to_allo': len(matched_true_to_allo),
    }


def compute_metrics_for_trainer_3class(eval_pred, tolerance=5):
    """Compute metrics for the trainer - 3-class version"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    all_predictions = predictions.flatten()
    all_labels = labels.flatten()

    mask = all_labels != -100
    all_predictions = all_predictions[mask]
    all_labels = all_labels[mask]

    accuracy = (all_predictions == all_labels).mean()
    switch_metrics = evaluate_switch_detection_with_proximity_3class(all_labels, all_predictions, tolerance=tolerance)

    return {
        'accuracy': float(accuracy),
        'switch_precision': float(switch_metrics['precision']),
        'switch_recall': float(switch_metrics['recall']),
        'switch_f1': float(switch_metrics['f1']),
        'true_switches': switch_metrics['true_switches'],
        'pred_switches': switch_metrics['pred_switches'],
        'exact_matches': switch_metrics['exact_matches'],
        'proximity_matches': switch_metrics['proximity_matches'],
        'to_auto_precision': float(switch_metrics['to_auto_precision']),
        'to_auto_recall': float(switch_metrics['to_auto_recall']),
        'to_allo_precision': float(switch_metrics['to_allo_precision']),
        'to_allo_recall': float(switch_metrics['to_allo_recall']),
        'matched_to_auto': switch_metrics['matched_to_auto'],
        'matched_to_allo': switch_metrics['matched_to_allo'],
    }


def apply_transition_constraints_3class(predictions):
    """Apply logical constraints to 3-class predictions"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    corrected_predictions = predictions.copy()
    current_mode = 0  # Start in Auto (inferred from context)

    for i in range(len(corrected_predictions)):
        pred = corrected_predictions[i]

        if pred == -100:
            continue

        # Switch→Auto (class 1) should follow allo mode
        if pred == 1:
            if current_mode != 1:  # Not in allo mode
                corrected_predictions[i] = 0  # Change to non-switch
            else:
                current_mode = 0  # Now in auto mode

        # Switch→Allo (class 2) should follow auto mode
        elif pred == 2:
            if current_mode != 0:  # Not in auto mode
                corrected_predictions[i] = 0  # Change to non-switch
            else:
                current_mode = 1  # Now in allo mode

        # For non-switch (class 0), mode stays as is

    return corrected_predictions


# ============================================================================
# 3-CLASS TRAINER
# ============================================================================

class SimpleSwitchTrainerAdditive3Class(Trainer):
    """3-class trainer with ADDITIVE loss - EXACT SAME logic"""

    def __init__(self, switch_recall_weight=10.0, proximity_tolerance=5,
                 segmentation_penalty=2.0, segmentation_reward=1.0,
                 proximity_reward=2.0, far_penalty=1.5,
                 apply_constraints=True, *args, **kwargs):

        # Handle processing_class/tokenizer
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')

        super().__init__(*args, **kwargs)

        # Initialize custom loss with EXACT SAME parameters
        self.custom_loss = SwitchFocusedLossAdditive3Class(
            switch_recall_weight=switch_recall_weight,
            proximity_tolerance=proximity_tolerance,
            segmentation_penalty=segmentation_penalty,
            segmentation_reward=segmentation_reward,
            proximity_reward=proximity_reward,
            far_penalty=far_penalty
        )

        self.apply_constraints = apply_constraints

        print(f"✓ 3-Class ADDITIVE Trainer initialized")
        print(f"  Switch recall weight: {switch_recall_weight}")
        print(f"  Proximity tolerance: {proximity_tolerance}")
        print(f"  Segmentation penalty: DISABLED")
        print(f"  Segmentation reward: DISABLED")
        print(f"  Proximity reward: {proximity_reward}")
        print(f"  Far penalty: {far_penalty}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation - EXACT SAME"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        seg_marks = inputs.get("seg_marks", None)
        loss = self.custom_loss(logits, labels, seg_marks)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Prediction with constraints - EXACT SAME logic"""
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
                    apply_transition_constraints_3class(predictions[b]),
                    device=predictions.device
                )

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_alto_3class_additive(model_name, output_dir):
    """
    Main training pipeline - 3-CLASS ADDITIVE version

    Args:
        model_name: Hugging Face model name or path
        output_dir: Directory to save the fine-tuned model
    """
    print("=" * 80)
    print("ALTO 3-CLASS with ADDITIVE LOSS")
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
    print("STEP 1: Initializing 3-class model")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Handle special tokenizers
    if model_name == 'sangjeedondrub/tibetan-roberta-base':
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            add_prefix_space=True
        )
    elif 'roberta' in model_name.lower():
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
        num_labels=3,  # ← 3 classes
        label2id={'non_switch': 0, 'to_auto': 1, 'to_allo': 2},
        id2label={0: 'non_switch', 1: 'to_auto', 2: 'to_allo'}
    )

    # Initialize biases (adjusted for 3 classes)
    with torch.no_grad():
        model.classifier.bias.data[0] = 0.0  # non-switch
        model.classifier.bias.data[1] = -1.0  # switch→auto
        model.classifier.bias.data[2] = -1.0  # switch→allo

    model = model.to(device)
    print(f"✓ Model initialized with 3 classes")

    # Create datasets (with automatic 4→3 class remapping)
    print("\n" + "=" * 80)
    print("STEP 2: Loading datasets (4-class → 3-class remapping)")
    print("=" * 80)

    train_dataset = CodeSwitchingDataset3Class(TRAIN_FILE, tokenizer)
    val_dataset = CodeSwitchingDataset3Class(VAL_FILE, tokenizer)
    test_dataset = CodeSwitchingDataset3Class(TEST_FILE, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    print(f"✓ Datasets loaded")

    # Training setup
    print("\n" + "=" * 80)
    print("STEP 3: Setting up training with 3-CLASS ADDITIVE loss")
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

    trainer = SimpleSwitchTrainerAdditive3Class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_for_trainer_3class(eval_pred, tolerance=5),
        switch_recall_weight=10.0,
        proximity_tolerance=5,
        segmentation_penalty=2.0,
        segmentation_reward=1.0,
        proximity_reward=2.0,
        far_penalty=1.5,
        apply_constraints=True
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    # Train
    print("\n" + "=" * 80)
    print("STEP 4: Training")
    print("=" * 80)
    print("Configuration:")
    print(f"  • 3 classes: Non-switch, Switch→Auto, Switch→Allo")
    print(f"  • 10 epochs")
    print(f"  • Learning rate: 2e-5")
    print(f"  • Batch size: 8")
    print(f"  • Switch recall weight: 10.0")
    print(f"  • Proximity tolerance: 5")
    print(f"  • Segmentation penalty: DISABLED")
    print(f"  • Segmentation reward: DISABLED")
    print(f"  • Proximity reward: 2.0")
    print(f"  • Far penalty: 1.5")
    print(f"  • Class weights: [0.1, 5.0, 5.0]")
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
    print(f"FINAL TEST RESULTS (3-CLASS)")
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
    print("STARTING 3-CLASS MULTI-MODEL TRAINING WITH ADDITIVE LOSS")
    print("=" * 80)
    print(f"\nTotal models to train: {len(MODELS_TO_TRAIN)}")

    all_results = []

    for idx, model_config in enumerate(MODELS_TO_TRAIN, 1):
        model_name = model_config['model_name']
        output_dir = model_config['output_dir']

        print("\n" + "=" * 80)
        print(f"TRAINING MODEL {idx}/{len(MODELS_TO_TRAIN)}")
        print("=" * 80)

        result = train_alto_3class_additive(
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
    print("3-CLASS TRAINING SUMMARY")
    print("=" * 80)
    for result in all_results:
        print(f"\nModel: {result['model_name']}")
        print(f"  Switch F1: {result['results']['eval_switch_f1']:.3f}")
        print(f"  Accuracy: {result['results']['eval_accuracy']:.3f}")
        print(f"  Output: {result['output_dir']}")

    print("\n" + "=" * 80)
    print("ALL 3-CLASS TRAINING COMPLETE!")
    print("=" * 80)