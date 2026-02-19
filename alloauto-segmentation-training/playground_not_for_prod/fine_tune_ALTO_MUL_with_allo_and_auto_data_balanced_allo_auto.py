"""
4-Class Code-Switching Detection System for Tibetan Text
Classes:
  0: Non-switching Auto (continuing in Auto mode)
  1: Non-switching Allo (continuing in Allo mode)
  2: Switch TO Auto
  3: Switch TO Allo

Focus: Proximity-aware loss and evaluation with 5-token tolerance
Modified to include additional data from allo and auto directories
BALANCED: Auto directory limited to same number of files as allo, only 'citation' files
"""
#MOST updated model! 18/9/2025"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import re
import docx
import os
import ssl
from typing import List, Tuple, Dict
from collections import defaultdict
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer
)
from torch.utils.data import Dataset
from pathlib import Path

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ssl._create_default_https_context = ssl._create_unverified_context

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Import all functions from the original module
from fine_tune_CS_4_classes_clean_no_allo_auto_labels_allow_non_switch_segments_no_25w_seq_overlap_seg_aware_loss import *


# ============================================================================
# NEW FUNCTIONS FOR PROCESSING ALLO AND AUTO DIRECTORIES
# ============================================================================

def aggressive_tag_removal(text):
    """
    Aggressively remove all tag patterns from text
    """
    if not text or not isinstance(text, str):
        return text

    # Remove all possible tag patterns
    patterns_to_remove = [
        r'<\s*[Aa][Ll][Ll][Oo]\s*>',  # <allo>, <ALLO>, < allo >, etc.
        r'<\s*[Aa][Uu][Tt][Oo]\s*>',  # <auto>, <AUTO>, < auto >, etc.
        r'<[Aa][Ll][Ll][Oo]',         # <allo, <ALLO without closing
        r'<[Aa][Uu][Tt][Oo]',         # <auto, <AUTO without closing
        r'[Aa][Ll][Ll][Oo]>',         # allo>, ALLO> without opening
        r'[Aa][Uu][Tt][Oo]>',         # auto>, AUTO> without opening
    ]

    cleaned_text = text
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, ' ', cleaned_text, flags=re.IGNORECASE)

    # Clean up multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text.strip()


def process_docx_with_tags(file_path, file_type="allo"):
    """
    Process a single .docx file that contains <allo> and <auto> tags
    Returns list of segments with tokens and labels
    """
    segments = []

    try:
        doc = docx.Document(file_path)
        full_text = ' '.join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

        if not full_text:
            return segments

        # Clean and normalize text
        full_text = clean_and_normalize_text(full_text)

        # Count words (excluding tags)
        text_without_tags = re.sub(r'<[^>]+>', '', full_text)
        word_count = len(text_without_tags.split())

        if word_count < 400:
            # Take entire file as one segment
            print(f"     File has {word_count} words (<400), using as single segment")
            tokens, labels = validate_preprocessing(full_text)

            if tokens:
                segment_id = f"{file_type}_{os.path.basename(file_path)}_{0:04d}"
                # Remove tags from original_text before storing
                original_text_clean = aggressive_tag_removal(full_text[:200])
                # Ensure tokens are clean strings without tags
                if isinstance(tokens, list):
                    tokens_str = ' '.join(tokens)
                else:
                    tokens_str = tokens
                tokens_str = aggressive_tag_removal(tokens_str)

                segments.append({
                    'segment_id': segment_id,
                    'source_file': os.path.basename(file_path),
                    'file_type': file_type,
                    'tokens': tokens_str,
                    'labels': labels,
                    'num_tokens': len(tokens) if isinstance(tokens, list) else len(tokens_str.split()),
                    'num_transitions': count_transitions_in_labels(labels),
                    'has_switch': has_switch_in_labels(labels),
                    'original_text': original_text_clean
                })
        else:
            # Process as usual, extract segments
            print(f"     File has {word_count} words, extracting segments")
            file_segments = extract_segments_with_token_limit(
                full_text,
                min_tokens=50,
                max_tokens=300,
                target_tokens=150
            )

            for i, seg in enumerate(file_segments):
                tokens, labels = validate_preprocessing(seg)

                if tokens:
                    segment_id = f"{file_type}_{os.path.basename(file_path)}_{i:04d}"
                    # Remove tags from original_text before storing
                    original_text_clean = aggressive_tag_removal(seg[:200])
                    # Ensure tokens are clean strings without tags
                    if isinstance(tokens, list):
                        tokens_str = ' '.join(tokens)
                    else:
                        tokens_str = tokens
                    tokens_str = aggressive_tag_removal(tokens_str)

                    segments.append({
                        'segment_id': segment_id,
                        'source_file': os.path.basename(file_path),
                        'file_type': file_type,
                        'tokens': tokens_str,
                        'labels': labels,
                        'num_tokens': len(tokens) if isinstance(tokens, list) else len(tokens_str.split()),
                        'num_transitions': count_transitions_in_labels(labels),
                        'has_switch': has_switch_in_labels(labels),
                        'original_text': original_text_clean
                    })

    except Exception as e:
        print(f"     Error processing {file_path}: {e}")

    return segments


def process_allo_directory(allo_dir):
    """
    Process all .docx files from allo directory
    """
    all_segments = []

    if not os.path.exists(allo_dir):
        print(f"⚠️ Allo directory does not exist: {allo_dir}")
        return all_segments, 0

    docx_files = [f for f in os.listdir(allo_dir) if f.endswith('.docx')]
    print(f"\n📁 Processing allo directory: {allo_dir}")
    print(f"   Found {len(docx_files)} .docx files")

    for file_name in docx_files:
        file_path = os.path.join(allo_dir, file_name)
        print(f"   Processing: {file_name}")

        segments = process_docx_with_tags(file_path, file_type="allo")
        all_segments.extend(segments)
        print(f"     ✅ Extracted {len(segments)} segments")

    print(f"   Total segments from allo dir: {len(all_segments)}")
    return all_segments, len(docx_files)  # Return segments AND file count


def process_auto_directory(auto_dir, max_files=None):
    """
    Process .docx files from auto directory
    Only processes files with 'citation' in the filename
    If max_files is specified, limits to that many files
    """
    all_segments = []

    if not os.path.exists(auto_dir):
        print(f"⚠️ Auto directory does not exist: {auto_dir}")
        return all_segments

    # Filter for .docx files with 'citation' in the name
    all_docx_files = [f for f in os.listdir(auto_dir) if f.endswith('.docx')]
    citation_files = [f for f in all_docx_files if 'citation' in f.lower()]

    print(f"\n📁 Processing auto directory: {auto_dir}")
    print(f"   Found {len(all_docx_files)} total .docx files")
    print(f"   Found {len(citation_files)} files with 'citation' in name")

    # Limit number of files if specified
    if max_files is not None and len(citation_files) > max_files:
        citation_files = citation_files[:max_files]
        print(f"   📊 BALANCING: Limiting to {max_files} files (to match allo directory)")

    print(f"   Will process {len(citation_files)} files")

    for file_name in citation_files:
        file_path = os.path.join(auto_dir, file_name)
        print(f"   Processing: {file_name}")

        segments = process_docx_with_tags(file_path, file_type="auto")
        all_segments.extend(segments)
        print(f"     ✅ Extracted {len(segments)} segments")

    print(f"   Total segments from auto dir: {len(all_segments)}")
    return all_segments


def count_transitions_in_labels(labels):
    """Count number of transitions in label sequence"""
    return sum(1 for label in labels if label in [2, 3])


def has_switch_in_labels(labels):
    """Check if labels contain any switch"""
    return any(label in [2, 3] for label in labels)


def deduplicate_segments(segments):
    """Remove duplicate segments based on tokens"""
    seen_tokens = set()
    unique_segments = []

    for segment in segments:
        # Convert tokens list to string for comparison
        if isinstance(segment['tokens'], list):
            tokens_str = ' '.join(segment['tokens'])
        else:
            tokens_str = segment['tokens']

        if tokens_str not in seen_tokens:
            seen_tokens.add(tokens_str)
            unique_segments.append(segment)

    if len(segments) != len(unique_segments):
        print(f"   🔍 Removed {len(segments) - len(unique_segments)} duplicate segments")

    return unique_segments


def format_segment_for_df(segment):
    """Helper function to format segment for DataFrame"""
    if isinstance(segment['tokens'], list):
        tokens_str = ' '.join(segment['tokens'])
    else:
        tokens_str = segment['tokens']

    # Ensure tokens are clean
    tokens_str = aggressive_tag_removal(tokens_str)

    if isinstance(segment['labels'], list):
        labels_str = ','.join(map(str, segment['labels']))
    else:
        labels_str = segment['labels']

    # Clean original_text too
    original_text_clean = aggressive_tag_removal(segment.get('original_text', ''))

    return {
        'segment_id': segment['segment_id'],
        'source_file': segment['source_file'],
        'file_type': segment['file_type'],
        'tokens': tokens_str,
        'labels': labels_str,
        'num_tokens': segment['num_tokens'],
        'num_transitions': segment['num_transitions'],
        'has_switch': segment['has_switch'],
        'original_text': original_text_clean
    }


def train_tibetan_code_switching_with_additional_data(
    target_switch_ratio=0.67,
    allo_dir='alloauto-segmentation-training/data/data_allo_orna/',
    auto_dir='alloauto-segmentation-training/data/data_auto_Orna/'
):
    """
    Modified training pipeline that includes additional data from allo and auto directories
    ONLY for train and val sets. Test set remains from original data only.
    Auto files are balanced to match allo file count, using only 'citation' files.
    """
    print("=" * 80)
    print("TIBETAN CODE-SWITCHING DETECTION TRAINING")
    print("With Additional Data Sources (Train/Val only)")
    print("BALANCED: Auto citation files limited to match allo file count")
    print("With Proximity-Aware Loss (5-token tolerance)")
    print("With Logical Transition Constraints")
    print("=" * 80)

    # Step 1: Process original files
    print("\n" + "=" * 60)
    print("STEP 1: Processing original files...")
    print("=" * 60)
    data_dir = 'dataset/annotated-data-raw'

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory {data_dir} not found!")
        return

    df_original, original_segments = process_all_files(data_dir, require_switch=False)
    print(f"Original segments: {len(original_segments)}")

    # Step 2: Balance ORIGINAL segments first
    print("\n" + "=" * 60)
    print("STEP 2: Balancing original segments...")
    print("=" * 60)
    balanced_original_segments = balance_segments_by_switches(original_segments, target_switch_ratio)

    # Convert to DataFrame for splitting
    segments_data_original = []
    for segment in balanced_original_segments:
        if isinstance(segment['tokens'], list):
            tokens_str = ' '.join(segment['tokens'])
        else:
            tokens_str = segment['tokens']

        # Ensure tokens are clean
        tokens_str = aggressive_tag_removal(tokens_str)

        if isinstance(segment['labels'], list):
            labels_str = ','.join(map(str, segment['labels']))
        else:
            labels_str = segment['labels']

        # Clean original_text
        original_text_clean = aggressive_tag_removal(segment.get('original_text', ''))

        segments_data_original.append({
            'segment_id': segment['segment_id'],
            'source_file': segment['source_file'],
            'file_type': segment['file_type'],
            'tokens': tokens_str,
            'labels': labels_str,
            'num_tokens': segment['num_tokens'],
            'num_transitions': segment['num_transitions'],
            'has_switch': segment['has_switch'],
            'original_text': original_text_clean
        })

    df_original = pd.DataFrame(segments_data_original)

    # Save original data
    df_original.to_csv('all_segments_original_300_400_tokens.csv', index=False)

    # Step 3: Create train/val/test split FROM ORIGINAL DATA ONLY
    print("\n" + "=" * 60)
    print("STEP 3: Creating train/val/test split from ORIGINAL data...")
    print("=" * 60)
    train_df_original, val_df_original, test_df = create_train_val_test_split(df_original)

    print(f"\nOriginal data splits:")
    print(f"   Train: {len(train_df_original)} segments")
    print(f"   Val: {len(val_df_original)} segments")
    print(f"   Test: {len(test_df)} segments (WILL REMAIN UNCHANGED)")

    # Step 4: Process allo directory and get file count
    print("\n" + "=" * 60)
    print("STEP 4: Processing allo directory for train/val...")
    print("=" * 60)
    allo_segments, allo_file_count = process_allo_directory(allo_dir)

    # Step 5: Process auto directory with BALANCED file count
    print("\n" + "=" * 60)
    print("STEP 5: Processing auto directory for train/val (BALANCED)...")
    print("=" * 60)
    print(f"⚖️ Balancing: Will process up to {allo_file_count} citation files from auto directory")
    auto_segments = process_auto_directory(auto_dir, max_files=allo_file_count)

    # Step 6: Split new data into train/val (85% train, 15% val)
    print("\n" + "=" * 60)
    print("STEP 6: Splitting new data for train/val...")
    print("=" * 60)

    # Process allo segments
    allo_train_segments = []
    allo_val_segments = []
    if allo_segments:
        # Deduplicate first
        allo_segments = deduplicate_segments(allo_segments)
        # Split 85% train, 15% val
        n_allo_val = max(1, int(len(allo_segments) * 0.15))
        allo_val_segments = allo_segments[:n_allo_val]
        allo_train_segments = allo_segments[n_allo_val:]
        print(f"   Allo: {len(allo_train_segments)} train, {len(allo_val_segments)} val")

    # Process auto segments
    auto_train_segments = []
    auto_val_segments = []
    if auto_segments:
        # Deduplicate first
        auto_segments = deduplicate_segments(auto_segments)
        # Split 85% train, 15% val
        n_auto_val = max(1, int(len(auto_segments) * 0.15))
        auto_val_segments = auto_segments[:n_auto_val]
        auto_train_segments = auto_segments[n_auto_val:]
        print(f"   Auto (citation files): {len(auto_train_segments)} train, {len(auto_val_segments)} val")

    # Step 7: Combine training and validation data
    print("\n" + "=" * 60)
    print("STEP 7: Combining train and val with new data...")
    print("=" * 60)

    # Combine TRAINING segments (original + new)
    all_train_segments = train_df_original.to_dict('records') + \
                         [format_segment_for_df(seg) for seg in allo_train_segments] + \
                         [format_segment_for_df(seg) for seg in auto_train_segments]

    # Combine VALIDATION segments (original + new)
    all_val_segments = val_df_original.to_dict('records') + \
                       [format_segment_for_df(seg) for seg in allo_val_segments] + \
                       [format_segment_for_df(seg) for seg in auto_val_segments]

    print(f"\n📊 Combined training data:")
    print(f"   Original train: {len(train_df_original)} segments")
    print(f"   Allo train: {len(allo_train_segments)} segments")
    print(f"   Auto train (citation): {len(auto_train_segments)} segments")
    print(f"   Total train: {len(all_train_segments)} segments")

    print(f"\n📊 Combined validation data:")
    print(f"   Original val: {len(val_df_original)} segments")
    print(f"   Allo val: {len(allo_val_segments)} segments")
    print(f"   Auto val (citation): {len(auto_val_segments)} segments")
    print(f"   Total val: {len(all_val_segments)} segments")

    print(f"\n⚖️ Balance check:")
    print(f"   Allo files processed: {allo_file_count}")
    print(f"   Auto citation files processed: {min(allo_file_count, len([f for f in os.listdir(auto_dir) if f.endswith('.docx') and 'citation' in f.lower()]))}")

    # Convert to DataFrames
    train_df = pd.DataFrame(all_train_segments)
    val_df = pd.DataFrame(all_val_segments)

    # Remove duplicates within train and val
    original_train_size = len(train_df)
    train_df = train_df.drop_duplicates(subset=['tokens'])
    if len(train_df) < original_train_size:
        print(f"   Removed {original_train_size - len(train_df)} duplicate segments from train")

    original_val_size = len(val_df)
    val_df = val_df.drop_duplicates(subset=['tokens'])
    if len(val_df) < original_val_size:
        print(f"   Removed {original_val_size - len(val_df)} duplicate segments from val")

    # Step 8: Remove 25+ token overlaps between train and test
    print("\n" + "=" * 60)
    print("STEP 8: Removing train-test overlaps...")
    print("=" * 60)
    train_df = remove_duplicate_sequences_from_train(train_df, test_df, min_duplicate_length=25)

    # Also check val-test overlap
    val_df = remove_duplicate_sequences_from_train(val_df, test_df, min_duplicate_length=25)

    # Verify balance after deduplication
    verify_split_balance(train_df, val_df, test_df, target_ratio=target_switch_ratio)

    # Validate each split for tags
    print("\nValidating splits for tags...")
    if not validate_no_tags_in_data(train_df, "TRAIN (Enhanced)"):
        return
    if not validate_no_tags_in_data(val_df, "VAL (Enhanced)"):
        return
    if not validate_no_tags_in_data(test_df, "TEST (Original only)"):
        return

    print("\n✅ ALL SPLITS VALIDATED - NO TAGS FOUND")

    # Save splits
    train_df.to_csv('train_segments.csv', index=False)
    val_df.to_csv('val_segments.csv', index=False)
    test_df.to_csv('test_segments.csv', index=False)

    print(f"\n📊 Final dataset sizes:")
    print(f"   Train: {len(train_df)} segments (enhanced with balanced allo + citation auto)")
    print(f"   Val: {len(val_df)} segments (enhanced with balanced allo + citation auto)")
    print(f"   Test: {len(test_df)} segments (ORIGINAL DATA ONLY)")

    # Step 9: Initialize model and tokenizer (SAME AS ORIGINAL)
    print("\n" + "=" * 60)
    print("STEP 9: Initializing model and tokenizer...")
    print("=" * 60)

    model_name = 'OMRIDRORI/mbert-tibetan-continual-wylie-final'

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Verify tokenizer works with Tibetan text
    test_tokens = tokenizer.tokenize("བོད་ཡིག་གི་ཚིག་གྲུབ་")
    print(f"Tokenizer test: 'བོད་ཡིག་གི་ཚིག་གྲུབ་' -> {test_tokens[:5]}...")

    # Initialize model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=4,
        ignore_mismatched_sizes=True
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✅ Model moved to GPU: {torch.cuda.get_device_name(0)}")

    # Create datasets
    train_dataset = CodeSwitchingDataset4Class('train_segments.csv', tokenizer)
    val_dataset = CodeSwitchingDataset4Class('val_segments.csv', tokenizer)
    test_dataset = CodeSwitchingDataset4Class('test_segments.csv', tokenizer)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Step 10: Setup training
    print("\n" + "=" * 60)
    print("STEP 10: Setting up training...")
    print("=" * 60)

    # Data collator for padding
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=512
    )

    # Output directory
    print(f' balanced_citation_ {target_switch_ratio:.0%}')
    output_dir = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_combined_data_with_allo_and_auto_balanced_22_10'

    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
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
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        gradient_accumulation_steps=1,
        label_smoothing_factor=0.0
    )

    # Custom trainer class
    class SimpleSwitchTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
                kwargs['processing_class'] = kwargs.get('tokenizer')
            super().__init__(*args, **kwargs)

            self.loss_fn = SwitchFocusedLoss(
                switch_recall_weight=10.0,
                proximity_tolerance=5,
                segmentation_penalty=3.0,
                segmentation_reward=0.3
            )
            self.tokenizer = kwargs.get('tokenizer') or kwargs.get('processing_class')

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            input_ids = inputs.get("input_ids").clone()
            outputs = model(**inputs)
            logits = outputs.get('logits')

            tokens_with_seg_info = self.analyze_segmentation_marks(input_ids)
            loss = self.loss_fn(logits, labels, tokens_with_seg_info)
            return (loss, outputs) if return_outputs else loss

        def analyze_segmentation_marks(self, input_ids):
            batch_size, seq_len = input_ids.shape
            seg_marks = torch.zeros_like(input_ids, dtype=torch.float)

            for b in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b])
                for t, token in enumerate(tokens):
                    if token and ('/' in token or '།' in token):
                        seg_marks[b, t] = 1.0

            return seg_marks

    # Initialize trainer
    trainer = SimpleSwitchTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_for_trainer(eval_pred, tolerance=5)
    )

    # Add early stopping callback
    from transformers import EarlyStoppingCallback
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    # Step 11: Train the model
    print("\n" + "=" * 60)
    print("STEP 11: Training model...")
    print("=" * 60)
    print("\nTraining with logical transition constraints...")
    print("Training data enhanced with:")
    print(f"  - Original: dataset/annotated-data-raw")
    print(f"  - Allo: {allo_dir} ({allo_file_count} files)")
    print(f"  - Auto (citation only): {auto_dir} (up to {allo_file_count} files)")
    print("Test data: ORIGINAL ONLY (unchanged)")

    trainer.train()

    # Save final model
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')

    # Step 12: Evaluation
    print("\n" + "=" * 60)
    print("STEP 12: Evaluating on test set (ORIGINAL DATA ONLY)...")
    print("=" * 60)

    # First, evaluate without constraints to see raw performance
    print("\nRaw performance (without constraints):")
    raw_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\n=== Raw Test Results (no constraints) ===")
    print(f"Switch F1: {raw_results['eval_switch_f1']:.3f}")
    print(f"Switch Precision: {raw_results['eval_switch_precision']:.3f}")
    print(f"Switch Recall: {raw_results['eval_switch_recall']:.3f}")

    # Now apply constraints during evaluation
    print("\n=== Final Test Results (with logical constraints) ===")
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Switch F1: {test_results['eval_switch_f1']:.3f}")
    print(f"Switch Precision: {test_results['eval_switch_precision']:.3f}")
    print(f"Switch Recall: {test_results['eval_switch_recall']:.3f}")
    print(f"Exact Matches: {test_results.get('eval_exact_matches', 0)}")
    print(f"Proximity Matches: {test_results.get('eval_proximity_matches', 0)}")
    print(f"True Switches: {test_results['eval_true_switches']}")
    print(f"Predicted Switches: {test_results['eval_pred_switches']}")

    # Per-type analysis
    print(f"\nPer-Type Performance:")
    print(f"  Switch→Auto Precision: {test_results.get('eval_to_auto_precision', 0):.3f}")
    print(f"  Switch→Auto Recall: {test_results.get('eval_to_auto_recall', 0):.3f}")
    print(f"  Switch→Allo Precision: {test_results.get('eval_to_allo_precision', 0):.3f}")
    print(f"  Switch→Allo Recall: {test_results.get('eval_to_allo_recall', 0):.3f}")

    # Check balance
    auto_count = test_results.get('eval_matched_to_auto', 0)
    allo_count = test_results.get('eval_matched_to_allo', 0)

    if allo_count == 0 and test_results.get('eval_true_to_allo', 0) > 0:
        print("\n⚠️ WARNING: Model still not predicting Switch→Allo!")
        print("  Consider: 1) More training epochs, 2) Higher weight for Switch→Allo")

    # Show some examples
    print("\nShowing test examples with logical constraints...")
    print_test_examples_with_constraints(model, tokenizer, 'test_segments.csv', num_examples=3, tolerance=5)

    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {output_dir}/final_model")
    print(f"\n📊 Data summary:")
    print(f"  Training: Enhanced with {len(allo_train_segments)} allo + {len(auto_train_segments)} auto citation segments")
    print(f"  Validation: Enhanced with {len(allo_val_segments)} allo + {len(auto_val_segments)} auto citation segments")
    print(f"  Data Balance: {allo_file_count} allo files ≈ {min(allo_file_count, len([f for f in os.listdir(auto_dir) if f.endswith('.docx') and 'citation' in f.lower()]))} auto citation files")
    print(f"  Test: ORIGINAL DATA ONLY (unchanged from base dataset)")
    print(f"\nLogical constraints: Auto→Allo and Allo→Auto only")

    return trainer, model, tokenizer, test_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test preprocessing first
    test_text = "<auto> བོད་ཡིག་གི་ <allo> ཚིག་གྲུབ་དང་ <auto> སྐད་ཡིག་"
    tokens, labels = validate_preprocessing(test_text)
    print(f"Test preprocessing: {len(tokens)} tokens, {len(labels)} labels")

    # Run training with additional data
    trainer, model, tokenizer, results = train_tibetan_code_switching_with_additional_data(
        target_switch_ratio=0.67,
        allo_dir='alloauto-segmentation-training/data/data_allo_orna/',
        auto_dir='alloauto-segmentation-training/data/data_auto_Orna/'
    )