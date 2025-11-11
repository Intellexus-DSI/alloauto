"""
4-Class ALTO with ADDITIVE Loss (Matching SimpleSwitchTrainer Behavior)
Converts multiplicative adjustments to additive adjustments

Original SimpleSwitchTrainer uses:
- Multiplicative: loss *= factor
New Additive version uses:
- Additive: loss += value

Uses existing CSV files: train_segments_clean.csv, val_segments.csv, test_segments.csv
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

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# ============================================================================
# DATASET CLASS (unchanged - reads 4-class data)
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
# ADDITIVE LOSS (Matching SimpleSwitchTrainer/SwitchFocusedLoss behavior)
# ============================================================================

class SwitchFocusedLossAdditive(nn.Module):
    """
    ADDITIVE version of SwitchFocusedLoss

    Original (Multiplicative):
    - loss *= 3.0 (penalty)
    - loss *= 0.3 (reward)
    - loss *= 0.1 (big reward)
    - loss *= 2.0 (mild penalty)

    New (Additive):
    - loss += 2.0 (penalty for seg misalignment)
    - loss -= 1.0 (reward for seg alignment)
    - loss -= 2.0 (big reward for proximity match)
    - loss += 1.5 (mild penalty for far predictions)
    """

    def __init__(self,
                 switch_recall_weight=10.0,  # Same as original
                 proximity_tolerance=5,  # Same as original
                 segmentation_penalty=2.0,  # Additive: add to loss
                 segmentation_reward=1.0,  # Additive: subtract from loss
                 proximity_reward=2.0,  # Additive: subtract for close match
                 far_penalty=1.5):  # Additive: add for far prediction
        super().__init__()
        self.proximity_tolerance = proximity_tolerance
        self.segmentation_penalty = segmentation_penalty
        self.segmentation_reward = segmentation_reward
        self.proximity_reward = proximity_reward
        self.far_penalty = far_penalty

        # Same class weights as original SwitchFocusedLoss
        self.class_weights = torch.tensor([
            0.1,  # Nearly ignore non-switch classes
            0.1,
            5.0,  # Focus on switches
            5.0
        ])

    def check_segmentation_alignment(self, seg_marks, predictions, b, seq_len):
        """
        Check if switches align with segmentation marks / or //
        Returns ADDITIVE adjustments (not multiplicative factors)

        Original:
        - Bad placement: *= 3.0
        - Good placement: *= 0.3

        New Additive:
        - Bad placement: += 2.0
        - Good placement: -= 1.0
        """
        seg_adjustments = torch.zeros(seq_len, device=predictions.device)

        for t in range(seq_len):
            if predictions[b, t] >= 2:  # This is a predicted switch

                # Check if switch happens 1-2 positions AFTER segmentation mark (BAD)
                for offset in [1, 2]:
                    if t - offset >= 0 and seg_marks[b, t - offset] > 0:
                        # Switch AFTER seg mark - PENALTY
                        seg_adjustments[t] += self.segmentation_penalty
                        break

                # Check if switch happens AT segmentation mark (GOOD)
                if seg_marks[b, t] > 0:
                    # Switch AT seg mark - REWARD
                    seg_adjustments[t] -= self.segmentation_reward

                # Check if switch happens RIGHT BEFORE segmentation mark (GOOD)
                elif t + 1 < seq_len and seg_marks[b, t + 1] > 0:
                    # Switch BEFORE seg mark - REWARD
                    seg_adjustments[t] -= self.segmentation_reward

        return seg_adjustments

    def forward(self, logits, labels, seg_marks=None):
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device
        self.class_weights = self.class_weights.to(device)

        # Base loss with heavy switch weighting (same as original)
        base_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='none'
        ).view(batch_size, seq_len)

        predictions = torch.argmax(logits, dim=-1)

        # ADDITIVE adjustments (replaces multiplicative)
        additive_adjustments = torch.zeros_like(base_loss)

        # Apply segmentation alignment if provided
        if seg_marks is not None:
            for b in range(batch_size):
                seg_adjustment = self.check_segmentation_alignment(seg_marks, predictions, b, seq_len)
                additive_adjustments[b] += seg_adjustment

        # Proximity logic with ADDITIVE adjustments
        for b in range(batch_size):
            true_switches = torch.where(labels[b] >= 2)[0]  # Both switch types
            pred_switches = torch.where(predictions[b] >= 2)[0]

            # REWARD predictions near true switches
            # Original: loss *= 0.1 (big reward)
            # New: loss -= 2.0 (subtract reward)
            for true_pos in true_switches:
                window_start = max(0, true_pos - self.proximity_tolerance)
                window_end = min(seq_len, true_pos + self.proximity_tolerance + 1)
                window_preds = predictions[b, window_start:window_end]

                if torch.any(window_preds >= 2):
                    # Prediction within tolerance window - BIG REWARD
                    additive_adjustments[b, true_pos] -= self.proximity_reward

            # Mild penalty for predictions far from any true switch
            # Original: loss *= 2.0
            # New: loss += 1.5
            for pred_pos in pred_switches:
                if len(true_switches) > 0:
                    distances = torch.abs(true_switches - pred_pos)
                    min_distance = torch.min(distances).item()

                    if min_distance > self.proximity_tolerance:
                        # Far from truth - MILD PENALTY
                        additive_adjustments[b, pred_pos] += self.far_penalty

        # Apply ADDITIVE adjustments to base loss
        adjusted_loss = base_loss + additive_adjustments

        valid_mask = (labels != -100).float()
        return (adjusted_loss * valid_mask).sum() / valid_mask.sum()


# ============================================================================
# EVALUATION FUNCTIONS (unchanged from original)
# ============================================================================

def evaluate_switch_detection_with_proximity(true_labels, pred_labels, tolerance=5):
    """Evaluate switch detection with proximity tolerance and TYPE matching"""
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Find switch positions BY TYPE
    true_switches_to_auto = np.where(true_labels == 2)[0]
    true_switches_to_allo = np.where(true_labels == 3)[0]
    pred_switches_to_auto = np.where(pred_labels == 2)[0]
    pred_switches_to_allo = np.where(pred_labels == 3)[0]

    matched_true_to_auto = set()
    matched_pred_to_auto = set()
    matched_true_to_allo = set()
    matched_pred_to_allo = set()

    exact_matches = 0
    proximity_matches = 0

    # Match Switch→Auto
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

    # Match Switch→Allo
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

    # Calculate metrics
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


def compute_metrics_for_trainer(eval_pred, tolerance=5):
    """Compute metrics for the trainer"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    all_predictions = predictions.flatten()
    all_labels = labels.flatten()

    mask = all_labels != -100
    all_predictions = all_predictions[mask]
    all_labels = all_labels[mask]

    accuracy = (all_predictions == all_labels).mean()
    switch_metrics = evaluate_switch_detection_with_proximity(all_labels, all_predictions, tolerance=tolerance)

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


def apply_transition_constraints(predictions, logits=None):
    """Apply logical constraints to predictions"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    corrected_predictions = predictions.copy()
    current_mode = 0  # Start in Auto

    for i in range(len(predictions)):
        pred = predictions[i]

        if pred == -100:
            continue

        if pred == 2:  # Switch to Auto
            if current_mode == 0:  # Already in Auto - INVALID
                corrected_predictions[i] = 0
            else:
                current_mode = 0

        elif pred == 3:  # Switch to Allo
            if current_mode == 1:  # Already in Allo - INVALID
                corrected_predictions[i] = 1
            else:
                current_mode = 1

        elif pred == 0:
            current_mode = 0

        elif pred == 1:
            current_mode = 1

    return corrected_predictions


# ============================================================================
# CUSTOM TRAINER (Matching SimpleSwitchTrainer but with ADDITIVE loss)
# ============================================================================

class SimpleSwitchTrainerAdditive(Trainer):
    """
    ADDITIVE version of SimpleSwitchTrainer
    Matches the original behavior but uses add/subtract instead of multiply
    """

    def __init__(self,
                 switch_recall_weight=10.0,
                 proximity_tolerance=5,
                 segmentation_penalty=2.0,
                 segmentation_reward=1.0,
                 proximity_reward=2.0,
                 far_penalty=1.5,
                 *args, **kwargs):
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.get('tokenizer')

        super().__init__(*args, **kwargs)

        # Use the ADDITIVE SwitchFocusedLoss
        self.loss_fn = SwitchFocusedLossAdditive(
            switch_recall_weight=switch_recall_weight,
            proximity_tolerance=proximity_tolerance,
            segmentation_penalty=segmentation_penalty,
            segmentation_reward=segmentation_reward,
            proximity_reward=proximity_reward,
            far_penalty=far_penalty
        )
        self.tokenizer = kwargs.get('tokenizer') or kwargs.get('processing_class')

    def analyze_segmentation_marks(self, input_ids):
        """
        Analyze tokens for Tibetan segmentation marks / and // and ། (Tibetan shad)
        Returns tensor marking positions with seg marks
        """
        batch_size, seq_len = input_ids.shape
        seg_marks = torch.zeros_like(input_ids, dtype=torch.float)

        for b in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b])
            for t, token in enumerate(tokens):
                if token and ('/' in token or '།' in token):
                    seg_marks[b, t] = 1.0

        return seg_marks

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_ids = inputs.get("input_ids").clone()
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Analyze segmentation marks
        seg_marks = self.analyze_segmentation_marks(input_ids)

        # Compute loss with segmentation awareness
        loss = self.loss_fn(logits, labels, seg_marks)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Apply transition constraints during evaluation"""
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        if not prediction_loss_only:
            loss, logits, labels = outputs
            predictions = torch.argmax(logits, dim=-1)

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

def train_alto_4class_additive():
    """
    Main training pipeline - ADDITIVE version of SimpleSwitchTrainer
    Uses existing CSV files
    """
    print("=" * 80)
    print("ALTO 4-CLASS with ADDITIVE LOSS (SimpleSwitchTrainer Style)")
    print("Using existing CSV files")
    print("=" * 80)

    # Use existing CSV files
    train_dataset_file = 'train_segments_clean.csv'
    val_dataset_file = 'val_segments.csv'
    test_dataset_file = 'test_segments.csv'

    # Check if files exist
    for f in [train_dataset_file, val_dataset_file, test_dataset_file]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found!")
            return

    print(f"\n✓ Using existing datasets:")
    print(f"  Train: {train_dataset_file}")
    print(f"  Val: {val_dataset_file}")
    print(f"  Test: {test_dataset_file}")

    # Initialize model
    print("\n" + "=" * 80)
    print("STEP 1: Initializing 4-class model")
    print("=" * 80)

    model_name = 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
    output_dir = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_4class_additive_loss_11_10'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    train_dataset = CodeSwitchingDataset4Class(train_dataset_file, tokenizer)
    val_dataset = CodeSwitchingDataset4Class(val_dataset_file, tokenizer)
    test_dataset = CodeSwitchingDataset4Class(test_dataset_file, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    print(f"✓ Datasets loaded")

    # Training setup
    print("\n" + "=" * 80)
    print("STEP 3: Setting up training with ADDITIVE loss")
    print("=" * 80)

    print("Conversion from Multiplicative to Additive:")
    print("  Original multiply by 3.0 → Add 2.0")
    print("  Original multiply by 0.3 → Subtract 1.0")
    print("  Original multiply by 0.1 → Subtract 2.0")
    print("  Original multiply by 2.0 → Add 1.5")

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
        push_to_hub=True,  # Enable pushing to HF
        hub_model_id="levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_MUL_loss_SAH",  # Your HF repo
    )

    # Create trainer with additive loss
    trainer = SimpleSwitchTrainerAdditive(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_for_trainer(eval_pred, tolerance=5),
        switch_recall_weight=10.0,  # Same as original
        proximity_tolerance=5,  # Same as original
        segmentation_penalty=2.0,  # Additive equivalent of *=3.0
        segmentation_reward=1.0,  # Additive equivalent of *=0.3
        proximity_reward=2.0,  # Additive equivalent of *=0.1
        far_penalty=1.5  # Additive equivalent of *=2.0
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    # Train
    print("\n" + "=" * 80)
    print("STEP 4: Training with additive SimpleSwitchTrainer")
    print("=" * 80)
    trainer.train()

    # Save
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')
    print(f"\n✓ Model saved to: {output_dir}/final_model")

    # Evaluate
    print("\n" + "=" * 80)
    print("STEP 5: Evaluating on test set")
    print("=" * 80)
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\n{'=' * 80}")
    print(f"FINAL TEST RESULTS (ADDITIVE SimpleSwitchTrainer)")
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
    print("STARTING ADDITIVE SimpleSwitchTrainer")
    print("=" * 80)
    print("\nMatching original SimpleSwitchTrainer/SwitchFocusedLoss behavior")
    print("But using ADDITIVE adjustments instead of MULTIPLICATIVE")
    print("\nMapping:")
    print("  Segmentation penalty: *=3.0 → +=2.0")
    print("  Segmentation reward: *=0.3 → -=1.0")
    print("  Proximity reward: *=0.1 → -=2.0")
    print("  Far penalty: *=2.0 → +=1.5")
    print("\nFeatures:")
    print("  ✓ Class weights: [0.1, 0.1, 5.0, 5.0] (same as original)")
    print("  ✓ Proximity tolerance: 5 tokens")
    print("  ✓ Segmentation awareness: / and //")
    print("  ✓ Transition constraints")

    trainer, model, tokenizer, results = train_alto_4class_additive()