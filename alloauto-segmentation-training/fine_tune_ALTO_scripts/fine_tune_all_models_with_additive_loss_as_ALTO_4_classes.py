"""
4-Class ALTO with ADDITIVE Loss (Multiple Models Version)
Configure models and output directories in arrays at the top
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
    # {   #ALTO additive -> ran this already!! 23/10
    #     'model_name': 'OMRIDRORI/mbert-tibetan-continual-wylie-final',
    #     'output_dir': './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_additive_loss_23_10'
    #     # 'output_dir': './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_additive_loss_10_10'
    # },
    # {   #CINO
    #     'model_name': 'hfl/cino-base-v2',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models/CINO_baseline_model_additive_loss_10_10'
    #
    # },
    # {   # XLM Roberta
    #     'model_name': 'xlm-roberta-base',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models/XLM_roberta_baseline_model_additive_loss_10_10'
    # },
    {   # mBERT
        'model_name': 'bert-base-multilingual-cased',
        'output_dir': './alloauto-segmentation-training/benchmark_models_ALTO_architecture/mBERT_vanilla_model_additive_loss_seg_23_10'
    },
    # {   # tibetian roberta
    #     'model_name': 'sangjeedondrub/tibetan-roberta-base',
    #     'output_dir': './alloauto-segmentation-training/benchmark_models/tibetan_roberta_baseline_model_additive_loss_10_10'
    # },
]

# Dataset files OLD
# TRAIN_FILE = 'train_segments_clean.csv'
# VAL_FILE = 'val_segments.csv'
# TEST_FILE = 'test_segments.csv'

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
# ADDITIVE LOSS
# ============================================================================

class SwitchFocusedLossAdditive(nn.Module):
    """ADDITIVE version of SwitchFocusedLoss"""

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

        self.class_weights = torch.tensor([
            0.1,  # Nearly ignore non-switch classes
            0.1,
            5.0,  # Focus on switches
            5.0
        ])

    def check_segmentation_alignment(self, seg_marks, predictions, b, seq_len):
        """Check if switches align with segmentation marks"""
        seg_adjustments = torch.zeros(seq_len, device=predictions.device)

        for t in range(seq_len):
            if predictions[b, t] >= 2:  # This is a predicted switch

                # Check if switch happens 1-2 positions AFTER segmentation mark (BAD)
                for offset in [1, 2]:
                    if t - offset >= 0 and seg_marks[b, t - offset] > 0:
                        seg_adjustments[t] += self.segmentation_penalty
                        break

                # Check if switch happens AT segmentation mark (GOOD)
                if seg_marks[b, t] > 0:
                    seg_adjustments[t] -= self.segmentation_reward

                # Check if switch happens RIGHT BEFORE segmentation mark (GOOD)
                elif t + 1 < seq_len and seg_marks[b, t + 1] > 0:
                    seg_adjustments[t] -= self.segmentation_reward

        return seg_adjustments

    def forward(self, logits, labels, seg_marks=None):
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

        # ADDITIVE adjustments
        additive_adjustments = torch.zeros_like(base_loss)

        # Apply segmentation alignment if provided
        if seg_marks is not None:
            for b in range(batch_size):
                seg_adjustment = self.check_segmentation_alignment(seg_marks, predictions, b, seq_len)
                additive_adjustments[b] += seg_adjustment

        # Proximity logic with ADDITIVE adjustments
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

        # Apply ADDITIVE adjustments to base loss
        adjusted_loss = base_loss + additive_adjustments

        valid_mask = (labels != -100).float()
        return (adjusted_loss * valid_mask).sum() / valid_mask.sum()


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_switch_detection_with_proximity(true_labels, pred_labels, tolerance=5):
    """Evaluate switch detection with proximity tolerance and TYPE matching"""
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

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
            if current_mode == 0:
                corrected_predictions[i] = 0
            else:
                current_mode = 0

        elif pred == 3:  # Switch to Allo
            if current_mode == 1:
                corrected_predictions[i] = 1
            else:
                current_mode = 1

        elif pred == 0:
            current_mode = 0

        elif pred == 1:
            current_mode = 1

    return corrected_predictions


# ============================================================================
# CUSTOM TRAINER
# ============================================================================

class SimpleSwitchTrainerAdditive(Trainer):
    """ADDITIVE version of SimpleSwitchTrainer"""

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
        """Analyze tokens for Tibetan segmentation marks"""
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

        seg_marks = self.analyze_segmentation_marks(input_ids)
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

def train_alto_4class_additive(model_name, output_dir):
    """
    Main training pipeline - ADDITIVE version of SimpleSwitchTrainer

    Args:
        model_name: Hugging Face model name or path
        output_dir: Directory to save the fine-tuned model
    """
    print("=" * 80)
    print("ALTO 4-CLASS with ADDITIVE LOSS (SimpleSwitchTrainer Style)")
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
    print("STEP 3: Setting up training with ADDITIVE loss")
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

    trainer = SimpleSwitchTrainerAdditive(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_for_trainer(eval_pred, tolerance=5),
        switch_recall_weight=10.0,
        proximity_tolerance=5,
        segmentation_penalty=2.0,
        segmentation_reward=1.0,
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
    print("STARTING MULTI-MODEL TRAINING")
    print("=" * 80)
    print(f"\nTotal models to train: {len(MODELS_TO_TRAIN)}")

    all_results = []

    for idx, model_config in enumerate(MODELS_TO_TRAIN, 1):
        model_name = model_config['model_name']
        output_dir = model_config['output_dir']

        print("\n" + "=" * 80)
        print(f"TRAINING MODEL {idx}/{len(MODELS_TO_TRAIN)}")
        print("=" * 80)

        result = train_alto_4class_additive(
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