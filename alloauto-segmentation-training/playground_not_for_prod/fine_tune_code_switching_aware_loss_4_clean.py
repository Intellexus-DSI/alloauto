"""
4-Class Code-Switching Detection System for Tibetan Text
Classes:
  0: Non-switching Auto (continuing in Auto mode)
  1: Non-switching Allo (continuing in Allo mode)
  2: Switch TO Auto
  3: Switch TO Allo
"""

import os
import ssl
import torch
from pathlib import Path

from fine_tune_code_switching_proximity_aware_loss_4_classes import process_mixed_files, stratified_split_combined_data
from fine_tune_code_switching_proximity_aware_loss_4_classes import CodeSwitchingDataset4Class
from fine_tune_code_switching_proximity_aware_loss_4_classes import ProximityAwareTrainer4Class
from fine_tune_code_switching_proximity_aware_loss_4_classes import evaluate_on_test_set_4class
from switch_detection_evaluation import evaluate_model_on_test_switch_detection

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification
)

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ssl._create_default_https_context = ssl._create_unverified_context


def check_data_balance(csv_path):
    """Check label distribution in dataset."""
    import pandas as pd
    from collections import Counter

    df = pd.read_csv(csv_path)
    all_labels = []
    for labels_str in df['labels']:
        all_labels.extend(list(map(int, labels_str.split(','))))

    label_counts = Counter(all_labels)
    total = len(all_labels)

    print(f"\nLabel distribution in {csv_path}:")
    label_names = ['Non-switch Auto', 'Non-switch Allo', 'Switch to Auto', 'Switch to Allo']
    for label in range(4):
        count = label_counts.get(label, 0)
        print(f"  {label_names[label]}: {count} ({count / total * 100:.2f}%)")

    switch_ratio = (label_counts.get(2, 0) + label_counts.get(3, 0)) / total
    print(f"Switch ratio: {switch_ratio:.4f}")

    return label_counts


def augment_training_data(train_csv, augmentation_factor=5):
    """Augment training data to balance classes."""
    import pandas as pd

    train_df = pd.read_csv(train_csv)
    switch_sequences = train_df[train_df['contains_switch'] == 1]
    no_switch_sequences = train_df[train_df['contains_switch'] == 0]

    print(f"\nOriginal training data:")
    print(f"  With switches: {len(switch_sequences)}")
    print(f"  Without switches: {len(no_switch_sequences)}")

    # Augment switch sequences
    augmented_switches = pd.concat([switch_sequences] * augmentation_factor)

    # Balance with no-switch sequences
    n_no_switch = min(len(no_switch_sequences), len(augmented_switches) * 2)
    balanced_no_switch = no_switch_sequences.sample(n=n_no_switch, random_state=42, replace=True)

    augmented_train = pd.concat([augmented_switches, balanced_no_switch])
    augmented_train = augmented_train.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = train_csv.replace('.csv', '_augmented.csv')
    augmented_train.to_csv(output_path, index=False)

    print(f"\nAugmented training data:")
    print(f"  Total sequences: {len(augmented_train)}")
    print(f"  With switches: {augmented_train['contains_switch'].sum()} "
          f"({augmented_train['contains_switch'].mean() * 100:.1f}%)")

    return output_path


def train_model_pipeline(data_dir='classify_allo_auto/data',
                         output_dir='classify_allo_auto/combined_data',
                         model_name='OMRIDRORI/mbert-tibetan-continual-unicode-240k',
                         output_model_dir='./classify_allo_auto/combined_model_4class'):
    """Main training pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Process all files
    print("=" * 60)
    print("STEP 1: Processing data files")
    print("=" * 60)

    combined_tokens, combined_sequences = process_mixed_files(
        data_dir=data_dir,
        output_dir=output_dir,
        sequence_length=512
    )

    # Step 2: Create train/val/test split
    print("\n" + "=" * 60)
    print("STEP 2: Creating train/val/test split")
    print("=" * 60)

    train_df, val_df, test_df = stratified_split_combined_data(
        combined_sequences_path=f'{output_dir}/all_sequences_combined.csv',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        ensure_file_diversity=True
    )

    # Step 3: Augment training data
    print("\n" + "=" * 60)
    print("STEP 3: Augmenting training data")
    print("=" * 60)

    train_augmented_path = augment_training_data('train_sequences_combined.csv')
    check_data_balance(train_augmented_path)

    # Step 4: Initialize model and train
    print("\n" + "=" * 60)
    print("STEP 4: Training model")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=4,
        label2id={
            'non_switch_auto': 0,
            'non_switch_allo': 1,
            'to_auto': 2,
            'to_allo': 3
        },
        id2label={
            0: 'non_switch_auto',
            1: 'non_switch_allo',
            2: 'to_auto',
            3: 'to_allo'
        }
    )

    # Initialize bias for rare switch classes
    with torch.no_grad():
        model.classifier.bias.data[2] = 5.0
        model.classifier.bias.data[3] = 5.0

    model = model.to(device)

    # Create datasets
    train_dataset = CodeSwitchingDataset4Class(train_augmented_path, tokenizer)
    val_dataset = CodeSwitchingDataset4Class('val_sequences_combined.csv', tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir=f'{output_model_dir}/logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='proximity_f1',
        greater_is_better=True,
        warmup_steps=1000,
        save_total_limit=2,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        push_to_hub=False,  # Set to True if you want to push to HuggingFace
    )

    # Initialize trainer with custom loss
    trainer = ProximityAwareTrainer4Class(
        model=model,
        args=training_args,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        max_distance=10,
        switch_loss_weight=50.0,
        false_positive_penalty=5.0
    )

    # Train
    trainer.train()

    # Save model
    final_model_path = f'{output_model_dir}/final_model'
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Step 5: Evaluate
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating on test set")
    print("=" * 60)

    test_metrics, test_results = evaluate_on_test_set_4class(
        model,
        tokenizer,
        'test_sequences_combined.csv',
        max_distance=10
    )

    # Switch detection evaluation
    switch_metrics, switch_results = evaluate_model_on_test_switch_detection(
        model,
        tokenizer,
        'test_sequences_combined.csv',
        tolerance=5,
        use_postprocess=True
    )

    print(f"\nFinal Results:")
    print(f"  Proximity F1: {test_metrics['proximity_f1']:.3f}")
    print(f"  Switch Detection F1: {switch_metrics['f1']:.3f}")

    return trainer, model, tokenizer, test_metrics, switch_metrics


if __name__ == "__main__":
    trainer, model, tokenizer, test_metrics, switch_metrics = train_model_pipeline()