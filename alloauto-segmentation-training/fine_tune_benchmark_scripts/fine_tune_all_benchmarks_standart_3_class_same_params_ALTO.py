"""
Multi-Model Training for Regular Baseline Architecture (Non-ALTO) - 3-CLASS VERSION
====================================================================================
This uses the simpler baseline approach from your XLM-RoBERTa code:
- Simple weighted cross-entropy loss (no segmentation awareness)
- No logical constraints
- Basic proximity evaluation (5-token tolerance)
- Same data as ALTO models for fair comparison

3-CLASS VERSION:
- Class 0: Non-switch (combines non_switch_auto + non_switch_allo)
- Class 1: Switch→Auto
- Class 2: Switch→Allo
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import gc
import json
import time
from datetime import datetime
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change as needed

# ============================================================================
# MODELS TO TRAIN (Same as ALTO for comparison)
# ============================================================================

MODELS_CONFIG = {
    'hfl/cino-base-v2': 'cino_base_v2_baseline_standard_weighted_3class_24_10_params_alto',
    'bert-base-multilingual-cased': 'mbert_cased_baseline_standard_weighted_3class_24_10_params_alto',
    'xlm-roberta-base': 'xlm_roberta_baseline_standard_weighted_3class_24_10_params_alto',
    'sangjeedondrub/tibetan-roberta-base': 'tibetan_roberta_baseline_standard_weighted_3class_24_10_params_alto',
    #
    'OMRIDRORI/mbert-tibetan-continual-wylie-final': 'mbert_tibetan_wylie_baseline_standard_weighted_3class_24_10_params_alto',
}

# Directories
DATA_DIR = 'dataset/preprocessed_augmented'  # Same augmented data as ALTO
OUTPUT_BASE_DIR = './alloauto-segmentation-training/benchmark_models_standard'


# ============================================================================
# LABEL REMAPPING: 4-CLASS TO 3-CLASS
# ============================================================================

def remap_4class_to_3class(label_4class):
    """
    Remap 4-class labels to 3-class:
    4-class: 0=non_switch_auto, 1=non_switch_allo, 2=switch→auto, 3=switch→allo
    3-class: 0=non_switch, 1=switch→auto, 2=switch→allo
    """
    if label_4class == -100:
        return -100
    elif label_4class in [0, 1]:  # Both non-switch types → class 0
        return 0
    elif label_4class == 2:  # Switch to auto → class 1
        return 1
    elif label_4class == 3:  # Switch to allo → class 2
        return 2
    else:
        return 0  # Default to non-switch


# ============================================================================
# YOUR EXACT BASELINE DATASET CLASS (WITH 3-CLASS REMAPPING)
# ============================================================================

class CodeSwitchingDataset(Dataset):
    """Your exact baseline dataset class."""

    def __init__(self, csv_file, tokenizer, max_length=512):
        print(f"  Loading data from: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"    Segments: {len(self.data)}")
        if 'source_file' in self.data.columns:
            print(f"    Files: {self.data['source_file'].nunique()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens'].split()
        labels_4class = list(map(int, row['labels'].split(',')))

        # Remap to 3-class
        labels = [remap_4class_to_3class(l) for l in labels_4class]

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
# YOUR EXACT WEIGHTED TRAINER (Simpler than ALTO)
# ============================================================================

class WeightedTrainer(Trainer):
    """Your exact baseline trainer with simple weighted loss - 3-CLASS VERSION."""

    def __init__(self, switch_weight=30.0, *args, **kwargs):
        # Handle tokenizer/processing_class
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.get('tokenizer')
        if 'tokenizer' in kwargs:
            kwargs.pop('tokenizer')

        super().__init__(*args, **kwargs)

        # 3-class weights (simple: low weight for non-switch, high for switches)
        self.class_weights = torch.tensor([
            1.0,  # Class 0: Non-switch (combined)
            switch_weight,  # Class 1: Switch to auto
            switch_weight  # Class 2: Switch to allo
        ])

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Simple weighted cross-entropy loss - NO ALTO features."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        device = logits.device
        self.class_weights = self.class_weights.to(device)

        batch_size, seq_len, num_classes = logits.shape

        # Simple weighted cross-entropy
        loss = nn.functional.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='mean',
            ignore_index=-100
        )

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# YOUR EXACT BASELINE METRICS
# ============================================================================

def evaluate_with_proximity(true_labels, pred_labels, tolerance=5):
    """Your exact baseline evaluation with proximity tolerance - 3-CLASS VERSION."""
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # 3-class: switches are classes 1 and 2 (instead of 2 and 3)
    true_auto = np.where(true_labels == 1)[0]
    true_allo = np.where(true_labels == 2)[0]
    pred_auto = np.where(pred_labels == 1)[0]
    pred_allo = np.where(pred_labels == 2)[0]

    matched_true = set()

    # Match predictions to true switches
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

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_switches': total_true,
        'pred_switches': total_pred,
        'matched': total_matched
    }


def compute_metrics(eval_pred):
    """Your exact baseline metrics computation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    all_preds = predictions.flatten()
    all_labels = labels.flatten()

    mask = all_labels != -100
    all_preds = all_preds[mask]
    all_labels = all_labels[mask]

    # Simple accuracy
    accuracy = (all_preds == all_labels).mean()

    # Proximity evaluation
    prox = evaluate_with_proximity(all_labels, all_preds, tolerance=5)

    return {
        'accuracy': float(accuracy),
        'precision': float(prox['precision']),
        'recall': float(prox['recall']),
        'f1': float(prox['f1']),
        'true_switches': prox['true_switches'],
        'pred_switches': prox['pred_switches'],
    }


# ============================================================================
# TRAINING FUNCTION FOR SINGLE MODEL
# ============================================================================

def train_baseline_model(model_name: str, save_name: str):
    """Train a single model with baseline architecture."""

    print(f"\n{'=' * 80}")
    print(f"TRAINING BASELINE: {model_name}")
    print(f"{'=' * 80}")
    print(f"Save name: {save_name}")

    output_dir = f'{OUTPUT_BASE_DIR}/{save_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Data paths - using same augmented data as ALTO
    train_path = f'{DATA_DIR}/train_segments_with_more_auto_allo.csv'
    val_path = f'{DATA_DIR}/val_segments_more_auto_allo.csv'
    test_path = f'{DATA_DIR}/test_segments_original.csv'

    # Check if clean train exists
    if os.path.exists(f'{DATA_DIR}/train_segments_clean.csv'):
        train_path = f'{DATA_DIR}/train_segments_clean.csv'
        print("  Using cleaned training data")

    try:
        # Load model and tokenizer
        print(f"\nLoading {model_name}...")
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
            num_labels=3,  # 3-CLASS VERSION
            label2id={'non_switch': 0, 'switch_to_auto': 1, 'switch_to_allo': 2},
            id2label={0: 'non_switch', 1: 'switch_to_auto', 2: 'switch_to_allo'}
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"  Device: {device}")

        # Create datasets
        print("\nLoading datasets:")
        train_dataset = CodeSwitchingDataset(train_path, tokenizer)
        val_dataset = CodeSwitchingDataset(val_path, tokenizer)
        test_dataset = CodeSwitchingDataset(test_path, tokenizer)

        data_collator = DataCollatorForTokenClassification(tokenizer)

        # YOUR EXACT BASELINE TRAINING ARGUMENTS
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=30,  # Your exact value
            save_strategy="steps",
            save_steps=60,  # Your exact value
            learning_rate=2e-5,  # Baseline uses 1e-5 (not 2e-5)
            # learning_rate=1e-5,  # Baseline uses 1e-5 (not 2e-5)
            per_device_train_batch_size=4,  # Baseline uses 4 (not 8)
            per_device_eval_batch_size=4,  # Baseline uses 4 (not 8)
            num_train_epochs=10,  # Baseline uses 10 (not 10)
            weight_decay=0.1,  # Baseline uses 0.1 (not 0.01)
            logging_steps=20,
            load_best_model_at_end=True,
            metric_for_best_model='f1',  # Baseline uses f1 (not switch_f1)
            greater_is_better=True,
            warmup_steps=200,  # Baseline uses 200 (not 100)
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to=[],
        )

        # Create baseline trainer
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            switch_weight=10.0  # Your exact baseline weight
            # switch_weight=30.0  # Your exact baseline weight
        )

        # Add early stopping
        # trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

        # Train
        print("\n" + "=" * 60)
        print("Training with BASELINE configuration:")
        print(f"  • 15 epochs")
        print(f"  • Learning rate: 1e-5")
        print(f"  • Batch size: 4")
        print(f"  • Switch weight: 30.0")
        print(f"  • Simple weighted CE loss (no ALTO features)")
        print(f"  • No logical constraints")
        print(f"  • No segmentation awareness")
        print("=" * 60)

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Save model
        trainer.save_model(f'{output_dir}/final_model')
        tokenizer.save_pretrained(f'{output_dir}/final_model')

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)

        print(f"\n=== {model_name} Baseline Results ===")
        print(f"Accuracy: {test_results.get('eval_accuracy', 0):.3f}")
        print(f"Precision: {test_results['eval_precision']:.3f}")
        print(f"Recall: {test_results['eval_recall']:.3f}")
        print(f"F1: {test_results['eval_f1']:.3f}")
        print(f"Training time: {training_time / 60:.1f} minutes")

        # Save results
        results = {
            'model_name': model_name,
            'save_name': save_name,
            'test_results': test_results,
            'training_time_seconds': training_time,
            'timestamp': datetime.now().isoformat()
        }

        with open(f'{output_dir}/results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return True, results

    except Exception as e:
        print(f"\n❌ Error training {model_name}: {str(e)}")
        return False, {'error': str(e)}

    finally:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_all_baseline_models():
    """Train all models with baseline architecture."""

    print("=" * 80)
    print("MULTI-MODEL BASELINE TRAINING (Non-ALTO)")
    print("=" * 80)
    print("Using simpler baseline architecture:")
    print("  • Simple weighted cross-entropy loss")
    print("  • No logical constraints")
    print("  • No segmentation awareness")
    print("  • Basic proximity evaluation (5-token)")
    print(f"\nModels to train: {len(MODELS_CONFIG)}")
    for model, save_name in MODELS_CONFIG.items():
        print(f"  • {model} → {save_name}")

    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Check data
    print(f"\nData directory: {DATA_DIR}")
    train_file = f'{DATA_DIR}/train_segments_with_more_auto_allo.csv'
    if not os.path.exists(train_file):
        print(f"❌ Training data not found: {train_file}")
        print("Please run augmentation script first.")
        return

    # Track results
    all_results = {}
    successful = []
    failed = []

    # Train each model
    for i, (model_name, save_name) in enumerate(MODELS_CONFIG.items(), 1):
        print(f"\n{'=' * 80}")
        print(f"MODEL {i}/{len(MODELS_CONFIG)}")
        print(f"{'=' * 80}")

        success, results = train_baseline_model(model_name, save_name)
        all_results[model_name] = results

        if success:
            successful.append(model_name)
        else:
            failed.append(model_name)

        # Save progress
        with open(f'{OUTPUT_BASE_DIR}/all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 80)

    print(f"\n✅ Successful: {len(successful)}/{len(MODELS_CONFIG)}")
    for model in successful:
        results = all_results[model]
        print(f"  • {MODELS_CONFIG[model]}")
        if 'test_results' in results:
            print(f"    F1: {results['test_results']['eval_f1']:.3f}")
            print(f"    Time: {results['training_time_seconds'] / 60:.1f} min")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for model in failed:
            print(f"  • {model}")

    print(f"\n📁 All models saved to: {OUTPUT_BASE_DIR}/")

    # Create comparison CSV
    if successful:
        comparison_data = []
        for model in successful:
            r = all_results[model]
            if 'test_results' in r:
                comparison_data.append({
                    'Model': MODELS_CONFIG[model],
                    'F1': f"{r['test_results']['eval_f1']:.3f}",
                    'Precision': f"{r['test_results']['eval_precision']:.3f}",
                    'Recall': f"{r['test_results']['eval_recall']:.3f}",
                    'Accuracy': f"{r['test_results'].get('eval_accuracy', 0):.3f}",
                    'Time (min)': f"{r['training_time_seconds'] / 60:.1f}"
                })

        if comparison_data:
            import pandas as pd
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(f'{OUTPUT_BASE_DIR}/baseline_comparison.csv', index=False)
            print(f"\n📊 Comparison saved to: {OUTPUT_BASE_DIR}/baseline_comparison.csv")
            print("\nBaseline Model Comparison:")
            print(comparison_df.to_string(index=False))

    return all_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check GPU
    print("GPU Status:")
    if torch.cuda.is_available():
        print(f"  ✅ {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️ No GPU available")

    # Train all baseline models
    results = train_all_baseline_models()