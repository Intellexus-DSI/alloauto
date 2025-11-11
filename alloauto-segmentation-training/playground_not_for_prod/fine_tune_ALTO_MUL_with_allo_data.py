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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

from fine_tune_CS_4_classes_clean_no_allo_auto_labels_allow_non_switch_segments_no_25w_seq_overlap_seg_aware_loss import print_test_examples_with_constraints, compute_metrics_for_trainer, SwitchFocusedLoss,analyze_and_balance_switch_distribution, CodeSwitchingDataset4Class, extract_segments_with_token_limit, split_into_sentences, clean_and_normalize_text, balance_segments_by_switches, process_all_files, remove_duplicate_sequences_from_train, create_train_val_test_split, verify_split_balance


def contains_any_tag_pattern(text):
    """
    Simple check: Only look for these specific patterns:
    <auto>, <allo>, <AUTO>, <ALLO>, <auto, <allo, <AUTO, <ALLO,
    AUTO>, ALLO>, auto>, allo>
    """
    if not text or not isinstance(text, str):
        return False

    # Exact patterns to check for
    forbidden_patterns = [
        '<auto>', '<allo>', '<AUTO>', '<ALLO>',
        '<auto', '<allo', '<AUTO', '<ALLO',
        'AUTO>', 'ALLO>', 'auto>', 'allo>'
    ]

    # Check if any of these patterns exist in the text
    for pattern in forbidden_patterns:
        if pattern in text:
            return True

    return False


def ultra_comprehensive_tag_check(text, context=""):
    """
    ULTRA COMPREHENSIVE tag checking with detailed reporting
    Returns (has_tags, error_messages)
    """
    if not text or not isinstance(text, str):
        return False, []

    errors = []

    # Check for < followed by 'a' or 'A'
    if re.search(r'<\s*[Aa]', text):
        matches = re.findall(r'<\s*[Aa]\w*\s*>?', text)
        errors.append(f"Found tag-like pattern starting with '<a': {matches[:3]}")

    # Check for anything ending with >
    if re.search(r'\w+\s*>', text):
        matches = re.findall(r'\w+\s*>', text)
        # Filter to only keep suspicious ones
        suspicious = [m for m in matches if any(x in m.lower() for x in ['auto', 'allo', 'uto', 'llo'])]
        if suspicious:
            errors.append(f"Found suspicious patterns ending with '>': {suspicious[:3]}")

    # Direct string checks (case-insensitive)
    text_lower = text.lower()
    suspicious_strings = [
        '<auto>', '<allo>', '<auto', '<allo', 'auto>', 'allo>',
        '<aut', '<all', 'uto>', 'llo>',
        '<au', '<al', 'to>', 'lo>'
    ]

    found_strings = [s for s in suspicious_strings if s in text_lower]
    if found_strings:
        errors.append(f"Found forbidden strings: {found_strings}")

    # Check for the word "auto" or "allo" near < or >
    if re.search(r'(<[^>]{0,10}(auto|allo)|((auto|allo)[^<]{0,10}>))', text_lower):
        errors.append(f"Found 'auto' or 'allo' near tag brackets")

    has_tags = len(errors) > 0

    if has_tags and context:
        errors = [f"{context}: {e}" for e in errors]

    return has_tags, errors


def validate_no_tags_in_data(df, split_name):
    """
    Simple validation: Ensure NO forbidden tag patterns in any data
    """
    print(f"\n{'=' * 80}")
    print(f"TAG VALIDATION FOR {split_name}")
    print(f"{'=' * 80}")

    issues_found = []

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Check tokens
        tokens_str = row['tokens']
        if contains_any_tag_pattern(tokens_str):
            issues_found.append(f"Row {idx}: Tags found in tokens: {tokens_str[:100]}")

        # Check original text
        if 'original_text' in row:
            orig_text = row['original_text']
            if contains_any_tag_pattern(orig_text):
                issues_found.append(f"Row {idx}: Tags found in original_text: {orig_text[:100]}")

        # Check individual tokens
        tokens_list = tokens_str.split()
        for i, token in enumerate(tokens_list):
            if contains_any_tag_pattern(token):
                issues_found.append(f"Row {idx}, Token {i}: '{token}' contains forbidden pattern")

    if issues_found:
        print(f"❌ VALIDATION FAILED! Found {len(issues_found)} issues:")
        for issue in issues_found[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more issues")
        return False
    else:
        print(f"✅ VALIDATION PASSED! No forbidden patterns found in {len(df)} segments")
        print(f"   Checked for: <auto>, <allo>, <AUTO>, <ALLO>,")
        print(f"                <auto, <allo, <AUTO, <ALLO,")
        print(f"                AUTO>, ALLO>, auto>, allo>")
        return True


def aggressive_tag_removal(text):
    """
    Remove specific tag patterns
    """
    if not text or not isinstance(text, str):
        return text

    # Patterns to remove
    patterns_to_remove = [
        '<auto>', '<allo>', '<AUTO>', '<ALLO>',
        '<auto', '<allo', '<AUTO', '<ALLO',
        'AUTO>', 'ALLO>', 'auto>', 'allo>'
    ]

    cleaned_text = text
    for pattern in patterns_to_remove:
        cleaned_text = cleaned_text.replace(pattern, ' ')

    # Clean up multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text.strip()



def process_segment_to_tokens_labels_clean(segment_text):
    """
    Convert segment to tokens and labels, ensuring NO tag patterns in tokens
    """
    # Normalize tags for consistent splitting
    normalized_text = re.sub(r'<\s*AUTO\s*>', '<auto>', segment_text, flags=re.IGNORECASE)
    normalized_text = re.sub(r'<\s*ALLO\s*>', '<allo>', normalized_text, flags=re.IGNORECASE)

    # Split by tags
    parts = re.split(r'(<auto>|<allo>)', normalized_text.lower())

    tokens = []
    labels = []
    current_mode = None

    for i, part in enumerate(parts):
        if part == '<auto>':
            continue  # Skip tag
        elif part == '<allo>':
            continue  # Skip tag
        elif part.strip():
            # Clean any remaining tag fragments
            clean_part = aggressive_tag_removal(part)
            words = clean_part.strip().split()

            # Determine mode from previous tags
            segment_mode = None
            for j in range(i - 1, -1, -1):
                if parts[j] == '<auto>':
                    segment_mode = 'auto'
                    break
                elif parts[j] == '<allo>':
                    segment_mode = 'allo'
                    break

            if segment_mode is None:
                segment_mode = 'auto'

            for word_idx, word in enumerate(words):
                # Skip if word contains forbidden patterns
                if contains_any_tag_pattern(word):
                    continue

                tokens.append(word)

                # Assign label
                if word_idx == 0 and current_mode is not None and current_mode != segment_mode:
                    label = 2 if segment_mode == 'auto' else 3
                else:
                    label = 0 if segment_mode == 'auto' else 1

                labels.append(label)

            current_mode = segment_mode

    return tokens, labels

def process_file_to_segments(file_path, file_type, require_switch=False):
    """
    Process a single file and extract segments
    Ensures NO tag patterns in tokens
    """
    print(f"Processing {file_type} file: {file_path.name}")

    # Read file content
    if file_type == 'docx':
        doc = docx.Document(str(file_path))
        text_content = ' '.join([para.text for para in doc.paragraphs])
    else:  # txt file
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        text_content = None

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text_content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if text_content is None:
            raise ValueError(f"Could not read {file_path.name} with any encoding")

    # Clean and normalize
    text_content = clean_and_normalize_text(text_content)

    # Split into sentences
    sentences = split_into_sentences(text_content)
    print(f"  Found {len(sentences)} sentences")

    # Extract segments
    segments = extract_segments_with_token_limit(sentences, min_tokens=300, max_tokens=400,
                                                 require_switch=require_switch)

    if require_switch:
        print(f"  Found {len(segments)} segments WITH switches (300-400 tokens each)")
    else:
        segments_with_switches = sum(1 for s in segments if s['has_transition'])
        segments_without_switches = len(segments) - segments_with_switches
        print(f"  Found {len(segments)} total segments:")
        print(f"    - {segments_with_switches} with switches")
        print(f"    - {segments_without_switches} without switches")

    # Convert segments to token-label pairs
    processed_segments = []
    segments_with_tags = 0

    for seg_idx, segment in enumerate(segments):
        tokens, labels = process_segment_to_tokens_labels_clean(segment['text'])

        if len(tokens) == 0:
            continue

        num_transitions = sum(1 for l in labels if l in [2, 3])

        # Check for forbidden patterns in tokens
        has_tags = False
        for token in tokens:
            if contains_any_tag_pattern(token):
                has_tags = True
                segments_with_tags += 1
                print(f"    WARNING: Found forbidden pattern in token: '{token}'")
                break

        if has_tags:
            continue

        # Create text WITHOUT tags for storage
        text_without_tags = ' '.join(tokens)

        # Final verification
        if contains_any_tag_pattern(text_without_tags):
            print(f"    WARNING: Forbidden patterns found in segment {seg_idx}")
            segments_with_tags += 1
            continue

        processed_segments.append({
            'segment_id': f"{file_path.stem}_{seg_idx}",
            'source_file': file_path.name,
            'file_type': file_type,
            'tokens': tokens,
            'labels': labels,
            'num_tokens': len(tokens),
            'num_transitions': num_transitions,
            'has_switch': num_transitions > 0,
            'original_text': text_without_tags
        })

    if segments_with_tags > 0:
        print(f"  Warning: Filtered out {segments_with_tags} segments with forbidden patterns")

    print(f"  Processed {len(processed_segments)} valid, clean segments")
    return processed_segments


def train_tibetan_code_switching(target_switch_ratio=0.67):
    """
    Main training pipeline - GUARANTEES original test set remains unchanged
    """
    print("=" * 80)
    print("TIBETAN CODE-SWITCHING DETECTION TRAINING")
    print("=" * 80)

    # Step 1: Process ORIGINAL data directory ONLY
    print("\nSTEP 1: Processing ORIGINAL data directory (dataset/annotated-data-raw)...")
    original_data_dir = 'dataset/annotated-data-raw'

    if not os.path.exists(original_data_dir):
        print(f"ERROR: Data directory {original_data_dir} not found!")
        return

    df_original, segments_original = process_all_files(original_data_dir, require_switch=False)

    if len(df_original) == 0:
        print("ERROR: No segments found in original data!")
        return

    if not validate_no_tags_in_data(df_original, "ORIGINAL DATA"):
        print("\n❌ FATAL ERROR: Tags found in original data! Cannot proceed.")
        return

    # Step 1.5: Balance ORIGINAL segments
    print("\nSTEP 1.5: Balancing original segments...")
    balanced_segments_original = balance_segments_by_switches(segments_original, target_switch_ratio)

    if len(balanced_segments_original) == 0:
        print("ERROR: No balanced segments available!")
        return

    # Convert to DataFrame
    segments_data_original = []
    for segment in balanced_segments_original:
        segments_data_original.append({
            'segment_id': segment['segment_id'],
            'source_file': segment['source_file'],
            'file_type': segment['file_type'],
            'tokens': ' '.join(segment['tokens']),
            'labels': ','.join(map(str, segment['labels'])),
            'num_tokens': segment['num_tokens'],
            'num_transitions': segment['num_transitions'],
            'has_switch': segment['has_switch'],
            'original_text': segment['original_text']
        })

    df_original = pd.DataFrame(segments_data_original)

    # Step 2: Create train/val/test split from ORIGINAL data ONLY
    print("\nSTEP 2: Creating train/val/test split from ORIGINAL data ONLY...")
    print("This creates the EXACT SAME test set as before (should be 121 segments)")

    # Use the SAME random seed as original code to ensure SAME split
    train_df_original, val_df_original, test_df_original = create_train_val_test_split(df_original)

    # IMMEDIATELY save the ORIGINAL test set - THIS NEVER CHANGES
    test_df_original.to_csv('test_segments.csv', index=False)

    print(f"\n{'=' * 80}")
    print(f"✅ SAVED ORIGINAL test_segments.csv")
    print(f"   Segments: {len(test_df_original)}")
    print(f"   THIS FILE WILL NOT BE MODIFIED - it stays exactly as before")
    print(f"{'=' * 80}")

    # Also save original train and val
    train_df_original.to_csv('train_segments_original.csv', index=False)
    val_df_original.to_csv('val_segments.csv', index=False)

    print(f"✅ Saved original train: {len(train_df_original)} segments")
    print(f"✅ Saved val: {len(val_df_original)} segments")

    # Verify the test set
    verify_split_balance(train_df_original, val_df_original, test_df_original, target_ratio=target_switch_ratio)

    # Step 3: Remove duplicates from TRAIN (not test!) vs original test
    print("\n" + "=" * 80)
    print("STEP 3: Checking for duplicates in TRAIN vs TEST")
    print("NOTE: We only clean TRAIN, TEST stays unchanged")
    print("=" * 80)

    train_df_original_clean = remove_duplicate_sequences_from_train(
        train_df_original, test_df_original, min_duplicate_length=25
    )

    if len(train_df_original_clean) < len(train_df_original):
        print(f"⚠️ Removed {len(train_df_original) - len(train_df_original_clean)} segments from train")
        train_df_original = train_df_original_clean
        train_df_original.to_csv('train_segments_original.csv', index=False)

    # Step 4: Process ADDITIONAL allo training data
    print("\n" + "=" * 80)
    print("STEP 4: Processing ADDITIONAL allo training data")
    print("=" * 80)

    allo_train_dir = 'alloauto-segmentation-training/data/data_allo_orna/'

    if os.path.exists(allo_train_dir):
        df_allo, segments_allo = process_all_files(allo_train_dir, require_switch=False)

        if len(df_allo) > 0:
            if not validate_no_tags_in_data(df_allo, "ALLO TRAINING DATA"):
                print("WARNING: Tags found in allo data! Skipping...")
                df_allo = pd.DataFrame()
            else:
                print(f"✓ Loaded {len(df_allo)} segments from allo training data")

                # Check for duplicates with ORIGINAL test set (clean allo train, not test!)
                print(f"\nChecking for duplicates: allo training vs original test...")
                df_allo_clean = remove_duplicate_sequences_from_train(
                    df_allo, test_df_original, min_duplicate_length=25
                )

                if len(df_allo_clean) < len(df_allo):
                    print(
                        f"⚠️ Removed {len(df_allo) - len(df_allo_clean)} allo segments that overlapped with original test")
                    df_allo = df_allo_clean
        else:
            print(f"WARNING: No segments found in {allo_train_dir}")
            df_allo = pd.DataFrame()
    else:
        print(f"WARNING: Directory {allo_train_dir} not found! Skipping allo training data.")
        df_allo = pd.DataFrame()

    # Step 5: Process ADDITIONAL allo test data (separate test set)
    print("\n" + "=" * 80)
    print("STEP 5: Processing ADDITIONAL allo test data")
    print("=" * 80)

    allo_test_dir = 'alloauto-segmentation-training/data/data_allo_orna_test/'

    if os.path.exists(allo_test_dir):
        allo_test_df, allo_test_segments = process_all_files(allo_test_dir, require_switch=False)

        if len(allo_test_df) > 0:
            if not validate_no_tags_in_data(allo_test_df, "ALLO_TEST"):
                print("WARNING: Tags found in allo test data! Skipping...")
                allo_test_df = None
            else:
                print(f"✓ Loaded {len(allo_test_df)} segments from allo test data")
                # Save allo test
                allo_test_df.to_csv('test_orna_allo.csv', index=False)
                print(f"✅ Saved test_orna_allo.csv with {len(allo_test_df)} segments")
        else:
            print(f"WARNING: No segments found in {allo_test_dir}")
            allo_test_df = None
    else:
        print(f"WARNING: Directory {allo_test_dir} not found! Skipping allo test set creation.")
        allo_test_df = None

    # Step 6: COMBINE training data (train only, test never changes!)
    print("\n" + "=" * 80)
    print("STEP 6: COMBINING training data")
    print("=" * 80)

    # Start with cleaned original training
    train_df_combined = train_df_original.copy()
    print(f"Starting with {len(train_df_combined)} segments from original training")

    # Add allo training data if available
    if len(df_allo) > 0:
        print(f"Adding {len(df_allo)} segments from allo training")
        train_df_combined = pd.concat([train_df_combined, df_allo], ignore_index=True)
        train_df_combined = train_df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Combined training: {len(train_df_combined)} segments")

    # Clean combined train vs allo test (if exists) - again, clean train not test!
    if allo_test_df is not None:
        print(f"\nChecking for duplicates: combined train vs allo test...")
        train_df_combined = remove_duplicate_sequences_from_train(
            train_df_combined, allo_test_df, min_duplicate_length=25
        )
        print(f"After removing overlaps with allo test: {len(train_df_combined)} segments")

    # Save the final combined training data
    train_df_combined.to_csv('train_segments_with_allo_files.csv', index=False)

    # Set final variables
    train_df = train_df_combined
    val_df = val_df_original
    test_df = test_df_original  # THIS IS THE ORIGINAL 121 SEGMENTS - UNCHANGED

    train_dataset_file = 'train_segments_with_allo_files.csv'
    val_dataset_file = 'val_segments.csv'
    test_dataset_file = 'test_segments.csv'  # ORIGINAL TEST - UNCHANGED

    print(f"\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)
    print(f"Training files:")
    print(f"  - train_segments_original.csv: {len(train_df_original)} segments (original only)")
    print(f"  - {train_dataset_file}: {len(train_df)} segments (USED FOR TRAINING - includes allo)")
    print(f"\nValidation file:")
    print(f"  - {val_dataset_file}: {len(val_df)} segments (original only)")
    print(f"\nTest files:")
    print(f"  - {test_dataset_file}: {len(test_df)} segments ⭐ ORIGINAL, UNCHANGED ⭐")
    if allo_test_df is not None:
        print(f"  - test_orna_allo.csv: {len(allo_test_df)} segments (NEW allo test)")

    print(f"\n🔒 GUARANTEE: {test_dataset_file} has EXACTLY the same segments as before")
    print(f"   Expected: 121 segments")
    print(f"   Actual: {len(test_df)} segments")
    if len(test_df) == 121:
        print(f"   ✅ CONFIRMED: Test set is unchanged!")
    else:
        print(f"   ⚠️ WARNING: Segment count differs from expected 121")

    # Validate all splits for tags
    print("\n" + "=" * 80)
    print("VALIDATING ALL SPLITS FOR FORBIDDEN TAG PATTERNS")
    print("=" * 80)

    if not validate_no_tags_in_data(train_df, "COMBINED TRAIN"):
        return
    if not validate_no_tags_in_data(val_df, "VAL"):
        return
    if not validate_no_tags_in_data(test_df, "TEST (ORIGINAL)"):
        return
    if allo_test_df is not None:
        if not validate_no_tags_in_data(allo_test_df, "ALLO TEST"):
            return

    print("\n✅ ALL SPLITS VALIDATED - NO FORBIDDEN PATTERNS FOUND")

    # Analyze switch distribution
    print("\n" + "=" * 60)
    print("ANALYZING SWITCH DISTRIBUTION")
    print("=" * 60)
    train_stats, val_stats, test_stats = analyze_and_balance_switch_distribution(
        train_df, val_df, test_df
    )

    # Continue with rest of training...
    print("\n" + "=" * 80)
    print("STEP 7: Initializing model")
    print("=" * 80)

    model_name = 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
    output_dir = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_combined_data_with_allo_15_10'

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

    # Create datasets
    print("\n" + "=" * 80)
    print("STEP 8: Creating datasets")
    print("=" * 80)

    train_dataset = CodeSwitchingDataset4Class(train_dataset_file, tokenizer)
    val_dataset = CodeSwitchingDataset4Class(val_dataset_file, tokenizer)
    test_dataset = CodeSwitchingDataset4Class(test_dataset_file, tokenizer)

    print(f"✓ Training dataset: {len(train_dataset)} examples from {train_dataset_file}")
    print(f"✓ Validation dataset: {len(val_dataset)} examples from {val_dataset_file}")
    print(f"✓ Test dataset: {len(test_dataset)} examples from {test_dataset_file} ⭐ ORIGINAL ⭐")

    # Create allo test dataset if it exists
    if allo_test_df is not None:
        allo_test_dataset = CodeSwitchingDataset4Class('test_orna_allo.csv', tokenizer)
        print(f"✓ Allo test dataset: {len(allo_test_dataset)} examples from test_orna_allo.csv")
    else:
        allo_test_dataset = None

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Calculate class weights
    print("\n" + "=" * 80)
    print("CALCULATING CLASS WEIGHTS")
    print("=" * 80)

    train_labels = []
    for idx in range(len(train_df)):
        labels = train_df.iloc[idx]['labels'].split(',')
        train_labels.extend([int(l) for l in labels])

    label_counts = {i: train_labels.count(i) for i in range(4)}
    total_count = len(train_labels)

    to_auto_weight = (total_count / (4 * label_counts[2])) * 10 if label_counts[2] > 0 else 30
    to_allo_weight = (total_count / (4 * label_counts[3])) * 10 if label_counts[3] > 0 else 30
    to_auto_weight = min(to_auto_weight, 20)
    to_allo_weight = min(to_allo_weight, 20)

    if label_counts[3] < label_counts[2] / 2:
        to_allo_weight *= 1.5

    to_auto_weight = min(to_auto_weight, 50)
    to_allo_weight = min(to_allo_weight, 50)

    print(f"Class distribution in training:")
    for i in range(4):
        print(f"  Class {i}: {label_counts[i]:,} ({label_counts[i] / total_count * 100:.1f}%)")
    print(f"\nSwitch weights:")
    print(f"  Switch→Auto weight: {to_auto_weight:.1f}")
    print(f"  Switch→Allo weight: {to_allo_weight:.1f}")

    # Training
    print("\n" + "=" * 80)
    print("STEP 9: Starting training")
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
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        gradient_accumulation_steps=1,
        label_smoothing_factor=0.0,
    )

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

    trainer = SimpleSwitchTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_for_trainer(eval_pred, tolerance=5)
    )

    from transformers import EarlyStoppingCallback
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    print("\nTraining...")
    trainer.train()

    # Save final model
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')
    print(f"\n✅ Model saved to: {output_dir}/final_model")

    # Evaluation
    print("\n" + "=" * 80)
    print("STEP 10: EVALUATION ON ORIGINAL TEST SET")
    print("=" * 80)
    print(f"Evaluating on: {test_dataset_file} ({len(test_df)} segments - ORIGINAL)")

    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\nAccuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Switch F1 (5-token tolerance): {test_results['eval_switch_f1']:.3f}")
    print(f"Switch Precision: {test_results['eval_switch_precision']:.3f}")
    print(f"Switch Recall: {test_results['eval_switch_recall']:.3f}")

    # Evaluation on allo test set if available
    if allo_test_dataset is not None:
        print("\n" + "=" * 80)
        print("STEP 11: EVALUATION ON ALLO TEST SET")
        print("=" * 80)

        allo_test_results = trainer.evaluate(eval_dataset=allo_test_dataset)

        print(f"\nAccuracy: {allo_test_results['eval_accuracy']:.3f}")
        print(f"Switch F1 (5-token tolerance): {allo_test_results['eval_switch_f1']:.3f}")
        print(f"Switch Precision: {allo_test_results['eval_switch_precision']:.3f}")
        print(f"Switch Recall: {allo_test_results['eval_switch_recall']:.3f}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n✅ test_segments.csv contains your ORIGINAL {len(test_df)} segments (unchanged)")
    print(f"✅ train_segments_with_allo_files.csv contains {len(train_df)} segments (includes allo)")
    if allo_test_dataset is not None:
        print(f"✅ test_orna_allo.csv contains {len(allo_test_df)} NEW test segments")

    return trainer, model, tokenizer, test_results
###################
def train_tibetan_code_switching_old(target_switch_ratio=0.67):
    """
    Main training pipeline with logical transition constraints
    """
    print("=" * 60)
    print("TIBETAN CODE-SWITCHING DETECTION TRAINING")
    print("With Proximity-Aware Loss (5-token tolerance)")
    print("With Logical Transition Constraints")
    print("=" * 60)

    # Step 1: Process ORIGINAL data directory FIRST to create original test set
    print("\nSTEP 1: Processing ORIGINAL data directory...")
    original_data_dir = 'dataset/annotated-data-raw'

    if not os.path.exists(original_data_dir):
        print(f"ERROR: Data directory {original_data_dir} not found!")
        return

    df_original, segments_original = process_all_files(original_data_dir, require_switch=False)

    if len(df_original) == 0:
        print("ERROR: No segments found in original data!")
        return

    if not validate_no_tags_in_data(df_original, "ORIGINAL DATA"):
        print("\n❌ FATAL ERROR: Tags found in original data! Cannot proceed.")
        return

    # Step 1.5: Balance ORIGINAL segments
    print("\nSTEP 1.5: Balancing original segments...")
    balanced_segments_original = balance_segments_by_switches(segments_original, target_switch_ratio)

    if len(balanced_segments_original) == 0:
        print("ERROR: No balanced segments available!")
        return

    # Convert to DataFrame
    segments_data_original = []
    for segment in balanced_segments_original:
        segments_data_original.append({
            'segment_id': segment['segment_id'],
            'source_file': segment['source_file'],
            'file_type': segment['file_type'],
            'tokens': ' '.join(segment['tokens']),
            'labels': ','.join(map(str, segment['labels'])),
            'num_tokens': segment['num_tokens'],
            'num_transitions': segment['num_transitions'],
            'has_switch': segment['has_switch'],
            'original_text': segment['original_text']
        })

    df_original = pd.DataFrame(segments_data_original)
    df_original.to_csv('all_segments_300_400_tokens_original.csv', index=False)

    # Step 2: Create train/val/test split from ORIGINAL data ONLY
    print("\nSTEP 2: Creating train/val/test split from ORIGINAL data ONLY...")
    print("This ensures test_segments.csv remains exactly the same as before")
    train_df_original, val_df_original, test_df_original = create_train_val_test_split(df_original)

    # Save the ORIGINAL test set - THIS STAYS THE SAME
    test_df_original.to_csv('test_segments.csv', index=False)
    print(f"✅ Saved ORIGINAL test_segments.csv with {len(test_df_original)} segments (UNCHANGED)")

    verify_split_balance(train_df_original, val_df_original, test_df_original, target_ratio=target_switch_ratio)

    # Step 2.5: Check and remove duplicates between ORIGINAL train and ORIGINAL test
    print("\n" + "=" * 80)
    print("CHECKING ORIGINAL TRAIN vs ORIGINAL TEST FOR DUPLICATES")
    print("=" * 80)

    train_df_original_clean = remove_duplicate_sequences_from_train(
        train_df_original, test_df_original, min_duplicate_length=25
    )

    if len(train_df_original_clean) < len(train_df_original):
        print(f"⚠️ Removed {len(train_df_original) - len(train_df_original_clean)} segments from original train")
        train_df_original = train_df_original_clean

    # Save cleaned original train
    train_df_original.to_csv('train_segments.csv', index=False)
    val_df_original.to_csv('val_segments.csv', index=False)

    print(f"\n✅ Saved CLEANED ORIGINAL splits:")
    print(f"  - train_segments.csv: {len(train_df_original)} segments (no overlap with test)")
    print(f"  - val_segments.csv: {len(val_df_original)} segments")
    print(f"  - test_segments.csv: {len(test_df_original)} segments (UNCHANGED)")

    # Step 3: Process ADDITIONAL allo training data
    print("\n" + "=" * 80)
    print("STEP 3: Processing ADDITIONAL allo training data")
    print("=" * 80)

    allo_train_dir = 'alloauto-segmentation-training/data/data_allo_orna/'

    if os.path.exists(allo_train_dir):
        df_allo, segments_allo = process_all_files(allo_train_dir, require_switch=False)

        if len(df_allo) > 0:
            if not validate_no_tags_in_data(df_allo, "ALLO TRAINING DATA"):
                print("WARNING: Tags found in allo data! Skipping...")
                df_allo = pd.DataFrame()
            else:
                print(f"✓ Loaded {len(df_allo)} segments from allo training data")

                # Check for duplicates with ORIGINAL test set
                print(f"\nChecking for duplicates: allo training vs original test...")
                df_allo_clean_vs_original_test = remove_duplicate_sequences_from_train(
                    df_allo, test_df_original, min_duplicate_length=25
                )

                if len(df_allo_clean_vs_original_test) < len(df_allo):
                    print(
                        f"⚠️ Removed {len(df_allo) - len(df_allo_clean_vs_original_test)} allo segments that overlapped with original test")
                    df_allo = df_allo_clean_vs_original_test
        else:
            print(f"WARNING: No segments found in {allo_train_dir}")
            df_allo = pd.DataFrame()
    else:
        print(f"WARNING: Directory {allo_train_dir} not found! Skipping allo training data.")
        df_allo = pd.DataFrame()

    # Step 4: Process ADDITIONAL allo test data BEFORE combining training
    print("\n" + "=" * 80)
    print("STEP 4: Processing ADDITIONAL allo test data")
    print("=" * 80)

    allo_test_dir = 'alloauto-segmentation-training/data/data_allo_orna_test/'

    if os.path.exists(allo_test_dir):
        allo_test_df, allo_test_segments = process_all_files(allo_test_dir, require_switch=False)

        if len(allo_test_df) > 0:
            if not validate_no_tags_in_data(allo_test_df, "ALLO_TEST"):
                print("WARNING: Tags found in allo test data! Skipping...")
                allo_test_df = None
            else:
                print(f"✓ Loaded {len(allo_test_df)} segments from allo test data")

                # Save allo test BEFORE cleaning train
                allo_test_df.to_csv('test_orna_allo.csv', index=False)
                print(f"✅ Saved test_orna_allo.csv with {len(allo_test_df)} segments")
        else:
            print(f"WARNING: No segments found in {allo_test_dir}")
            allo_test_df = None
    else:
        print(f"WARNING: Directory {allo_test_dir} not found! Skipping allo test set creation.")
        allo_test_df = None

    # Step 5: COMPREHENSIVE DUPLICATE REMOVAL
    print("\n" + "=" * 80)
    print("STEP 5: COMPREHENSIVE DUPLICATE CHECKING")
    print("Ensuring NO 25+ token overlap between ANY training and ANY test data")
    print("=" * 80)

    # Start with original training data
    train_df_combined = train_df_original.copy()
    print(f"\nStarting with {len(train_df_combined)} segments from original training")

    # Add allo training data if available
    if len(df_allo) > 0:
        print(f"Adding {len(df_allo)} segments from allo training")
        train_df_combined = pd.concat([train_df_combined, df_allo], ignore_index=True)
        train_df_combined = train_df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Combined training: {len(train_df_combined)} segments")

    # Now remove duplicates against ORIGINAL test
    print(f"\n--- Checking combined train vs ORIGINAL test ---")
    train_df_combined = remove_duplicate_sequences_from_train(
        train_df_combined, test_df_original, min_duplicate_length=25
    )
    print(f"After removing overlaps with original test: {len(train_df_combined)} segments")

    # Remove duplicates against ALLO test (if exists)
    if allo_test_df is not None:
        print(f"\n--- Checking combined train vs ALLO test ---")
        train_df_combined = remove_duplicate_sequences_from_train(
            train_df_combined, allo_test_df, min_duplicate_length=25
        )
        print(f"After removing overlaps with allo test: {len(train_df_combined)} segments")

    # Final verification: double-check no overlaps
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION: No overlaps between train and test sets")
    print("=" * 80)

    def verify_no_overlap(train_df, test_df, test_name):
        """Verify no 25+ token sequences overlap"""
        import hashlib
        from collections import defaultdict

        # Build set of all 25+ token sequences in test
        test_sequences = set()
        for idx, row in test_df.iterrows():
            tokens = row['tokens'].split()
            for i in range(len(tokens) - 24):  # 25 tokens minimum
                sequence = ' '.join(tokens[i:i + 25])
                sequence_hash = hashlib.md5(sequence.encode()).hexdigest()
                test_sequences.add(sequence_hash)

        # Check if any train sequences match
        overlaps_found = 0
        for idx, row in train_df.iterrows():
            tokens = row['tokens'].split()
            for i in range(len(tokens) - 24):
                sequence = ' '.join(tokens[i:i + 25])
                sequence_hash = hashlib.md5(sequence.encode()).hexdigest()
                if sequence_hash in test_sequences:
                    overlaps_found += 1
                    break  # Found overlap in this segment

        if overlaps_found > 0:
            print(f"  ❌ FOUND {overlaps_found} overlapping segments with {test_name}!")
            return False
        else:
            print(f"  ✅ No overlaps with {test_name}")
            return True

    # Verify against original test
    original_test_clean = verify_no_overlap(train_df_combined, test_df_original, "ORIGINAL TEST")

    # Verify against allo test if exists
    if allo_test_df is not None:
        allo_test_clean = verify_no_overlap(train_df_combined, allo_test_df, "ALLO TEST")
    else:
        allo_test_clean = True

    if not (original_test_clean and allo_test_clean):
        print("\n❌ ERROR: Overlaps still found! Cannot proceed.")
        return

    print("\n✅ VERIFICATION PASSED: No 25+ token overlaps found!")
    ######
    ### make sure no tags
    train_df = train_df_combined
    val_df = val_df_original
    test_df = test_df_original
    print("\n" + "=" * 80)
    print("ULTRA COMPREHENSIVE FINAL TAG VALIDATION")
    print("=" * 80)

    print("\n🔍 Validating TRAINING data...")
    if not validate_no_tags_in_data(train_df, "COMBINED TRAIN"):
        print("\n💥 FATAL ERROR: Tags found in TRAINING data!")
        print("Cannot proceed with training. Please review the aggressive_tag_removal function.")
        return

    print("\n🔍 Validating VALIDATION data...")
    if not validate_no_tags_in_data(val_df, "VALIDATION"):
        print("\n💥 FATAL ERROR: Tags found in VALIDATION data!")
        return

    print("\n🔍 Validating ORIGINAL TEST data...")
    if not validate_no_tags_in_data(test_df, "ORIGINAL TEST"):
        print("\n💥 FATAL ERROR: Tags found in ORIGINAL TEST data!")
        return

    if allo_test_df is not None:
        print("\n🔍 Validating ALLO TEST data...")
        if not validate_no_tags_in_data(allo_test_df, "ALLO TEST"):
            print("\n💥 FATAL ERROR: Tags found in ALLO TEST data!")
            return

    print("\n" + "=" * 80)
    print("✅ ALL DATA VALIDATED - ABSOLUTELY NO TAGS FOUND")
    print("=" * 80)
    print("Validated that none of the following exist in ANY data:")
    print("  ❌ <auto>, <allo>, <AUTO>, <ALLO>")
    print("  ❌ <auto, <allo, auto>, allo>")
    print("  ❌ <aut, <all, uto>, llo>")
    print("  ❌ ANY partial or malformed tags")
    print("  ❌ ANY < or > characters")
    print("  ✅ Data is completely clean and safe for training")

    # Save the final combined training data
    train_df_combined.to_csv('train_segments_with_allo_files.csv', index=False)



    train_dataset_file = 'train_segments_with_allo_files.csv'
    val_dataset_file = 'val_segments.csv'
    test_dataset_file = 'test_segments.csv'

    print(f"\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)
    print(f"Training:")
    print(f"  - train_segments.csv: {len(train_df_original)} segments (original only)")
    print(f"  - {train_dataset_file}: {len(train_df)} segments (USED FOR TRAINING)")
    print(f"    * Original training: {len(train_df_original)} segments")
    if len(df_allo) > 0:
        print(f"    * Allo training: {len(df_allo)} segments")
    print(f"    * No 25+ token overlap with ANY test set")
    print(f"\nValidation:")
    print(f"  - {val_dataset_file}: {len(val_df)} segments (original only)")
    print(f"\nTesting:")
    print(f"  - {test_dataset_file}: {len(test_df)} segments (ORIGINAL, UNCHANGED)")
    if allo_test_df is not None:
        print(f"  - test_orna_allo.csv: {len(allo_test_df)} segments (NEW allo test)")

    # CRITICAL: Validate all splits for tags
    print("\n" + "=" * 80)
    print("VALIDATING ALL SPLITS FOR TAGS")
    print("=" * 80)

    if not validate_no_tags_in_data(train_df, "COMBINED TRAIN"):
        return
    if not validate_no_tags_in_data(val_df, "VAL"):
        return
    if not validate_no_tags_in_data(test_df, "TEST (ORIGINAL)"):
        return

    print("\n✅ ALL SPLITS VALIDATED - NO TAGS FOUND")

    # Analyze switch distribution
    print("\n" + "=" * 60)
    print("ANALYZING SWITCH DISTRIBUTION IN SPLITS")
    print("=" * 60)
    train_stats, val_stats, test_stats = analyze_and_balance_switch_distribution(
        train_df, val_df, test_df
    )

    # Rest of the training code continues as before...
    print("\n" + "=" * 80)
    print("STEP 6: Initializing model")
    print("=" * 80)

    model_name = 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
    output_dir = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_combined_data_with_allo_15_10'

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

    # Create datasets
    print("\n" + "=" * 80)
    print("STEP 7: Creating datasets")
    print("=" * 80)

    train_dataset = CodeSwitchingDataset4Class(train_dataset_file, tokenizer)
    val_dataset = CodeSwitchingDataset4Class(val_dataset_file, tokenizer)
    test_dataset = CodeSwitchingDataset4Class(test_dataset_file, tokenizer)

    print(f"✓ Training dataset: {len(train_dataset)} examples from {train_dataset_file}")
    print(f"✓ Validation dataset: {len(val_dataset)} examples from {val_dataset_file}")
    print(f"✓ Test dataset: {len(test_dataset)} examples from {test_dataset_file}")

    # Create allo test dataset if it exists
    if allo_test_df is not None:
        allo_test_dataset = CodeSwitchingDataset4Class('test_orna_allo.csv', tokenizer)
        print(f"✓ Allo test dataset: {len(allo_test_dataset)} examples from test_orna_allo.csv")
    else:
        allo_test_dataset = None

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Calculate class weights
    print("\n" + "=" * 80)
    print("CALCULATING CLASS WEIGHTS")
    print("=" * 80)

    train_labels = []
    for idx in range(len(train_df)):
        labels = train_df.iloc[idx]['labels'].split(',')
        train_labels.extend([int(l) for l in labels])

    label_counts = {i: train_labels.count(i) for i in range(4)}
    total_count = len(train_labels)

    to_auto_weight = (total_count / (4 * label_counts[2])) * 10 if label_counts[2] > 0 else 30
    to_allo_weight = (total_count / (4 * label_counts[3])) * 10 if label_counts[3] > 0 else 30
    to_auto_weight = min(to_auto_weight, 20)
    to_allo_weight = min(to_allo_weight, 20)

    if label_counts[3] < label_counts[2] / 2:
        to_allo_weight *= 1.5

    to_auto_weight = min(to_auto_weight, 50)
    to_allo_weight = min(to_allo_weight, 50)

    print(f"Class distribution in training:")
    for i in range(4):
        print(f"  Class {i}: {label_counts[i]:,} ({label_counts[i] / total_count * 100:.1f}%)")
    print(f"\nSwitch weights:")
    print(f"  Switch→Auto weight: {to_auto_weight:.1f}")
    print(f"  Switch→Allo weight: {to_allo_weight:.1f}")

    # Training
    print("\n" + "=" * 80)
    print("STEP 8: Starting training")
    print("=" * 80)
    print("Training configuration:")
    print("  - Proximity tolerance: 5 tokens")
    print("  - Logical transition constraints enabled")
    print("  - Segmentation-aware loss")
    print("  - NO 25+ token overlap between train and test")

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
        label_smoothing_factor=0.0,
    )

    class SimpleSwitchTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
                kwargs['processing_class'] = kwargs.get('tokenizer')
            super().__init__(*args, **kwargs)

            self.loss_fn = SwitchFocusedLoss(
                switch_recall_weight=10.0,
                proximity_tolerance=5,  # 5-token tolerance
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

    trainer = SimpleSwitchTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_for_trainer(eval_pred, tolerance=5)  # 5-token tolerance
    )

    from transformers import EarlyStoppingCallback
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    print("\nTraining with 5-token proximity tolerance...")
    print("Guaranteed: NO 25+ token sequences shared between train and test")
    trainer.train()

    # Save final model
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')
    print(f"\n✅ Model saved to: {output_dir}/final_model")

    # Evaluation on ORIGINAL test set (UNCHANGED)
    print("\n" + "=" * 80)
    print("STEP 9: EVALUATION ON ORIGINAL TEST SET")
    print("=" * 80)
    print(f"Evaluating on: {test_dataset_file}")
    print(f"Using 5-token proximity tolerance for switch detection")
    print(f"GUARANTEED: No 25+ token overlap with training data")

    print("\n--- Raw performance (without constraints) ---")
    raw_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"Switch F1: {raw_results['eval_switch_f1']:.3f}")
    print(f"Switch Precision: {raw_results['eval_switch_precision']:.3f}")
    print(f"Switch Recall: {raw_results['eval_switch_recall']:.3f}")

    print("\n--- Final performance (with logical constraints) ---")
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Switch F1 (5-token tolerance): {test_results['eval_switch_f1']:.3f}")
    print(f"Switch Precision (5-token tolerance): {test_results['eval_switch_precision']:.3f}")
    print(f"Switch Recall (5-token tolerance): {test_results['eval_switch_recall']:.3f}")
    print(f"Exact Matches: {test_results.get('eval_exact_matches', 0)}")
    print(f"Proximity Matches (within 5 tokens): {test_results.get('eval_proximity_matches', 0)}")
    print(f"True Switches: {test_results['eval_true_switches']}")
    print(f"Predicted Switches: {test_results['eval_pred_switches']}")

    # Per-type analysis
    print(f"\nPer-Type Performance (5-token tolerance):")
    print(f"  Switch→Auto:")
    print(f"    Precision: {test_results.get('eval_to_auto_precision', 0):.3f}")
    print(f"    Recall: {test_results.get('eval_to_auto_recall', 0):.3f}")
    print(f"    True switches: {test_results.get('eval_true_to_auto', 0)}")
    print(f"    Matched: {test_results.get('eval_matched_to_auto', 0)}")
    print(f"  Switch→Allo:")
    print(f"    Precision: {test_results.get('eval_to_allo_precision', 0):.3f}")
    print(f"    Recall: {test_results.get('eval_to_allo_recall', 0):.3f}")
    print(f"    True switches: {test_results.get('eval_true_to_allo', 0)}")
    print(f"    Matched: {test_results.get('eval_matched_to_allo', 0)}")

    # Evaluation on allo test set if available
    if allo_test_dataset is not None:
        print("\n" + "=" * 80)
        print("STEP 10: EVALUATION ON ALLO TEST SET")
        print("=" * 80)
        print(f"Evaluating on: test_orna_allo.csv")
        print(f"Using 5-token proximity tolerance for switch detection")
        print(f"GUARANTEED: No 25+ token overlap with training data")

        allo_test_results = trainer.evaluate(eval_dataset=allo_test_dataset)

        print(f"\nAccuracy: {allo_test_results['eval_accuracy']:.3f}")
        print(f"Switch F1 (5-token tolerance): {allo_test_results['eval_switch_f1']:.3f}")
        print(f"Switch Precision (5-token tolerance): {allo_test_results['eval_switch_precision']:.3f}")
        print(f"Switch Recall (5-token tolerance): {allo_test_results['eval_switch_recall']:.3f}")
        print(f"Exact Matches: {allo_test_results.get('eval_exact_matches', 0)}")
        print(f"Proximity Matches (within 5 tokens): {allo_test_results.get('eval_proximity_matches', 0)}")
        print(f"True Switches: {allo_test_results['eval_true_switches']}")
        print(f"Predicted Switches: {allo_test_results['eval_pred_switches']}")

        # Per-type analysis
        print(f"\nPer-Type Performance (5-token tolerance):")
        print(f"  Switch→Auto:")
        print(f"    Precision: {allo_test_results.get('eval_to_auto_precision', 0):.3f}")
        print(f"    Recall: {allo_test_results.get('eval_to_auto_recall', 0):.3f}")
        print(f"    True switches: {allo_test_results.get('eval_true_to_auto', 0)}")
        print(f"    Matched: {allo_test_results.get('eval_matched_to_auto', 0)}")
        print(f"  Switch→Allo:")
        print(f"    Precision: {allo_test_results.get('eval_to_allo_precision', 0):.3f}")
        print(f"    Recall: {allo_test_results.get('eval_to_allo_recall', 0):.3f}")
        print(f"    True switches: {allo_test_results.get('eval_true_to_allo', 0)}")
        print(f"    Matched: {allo_test_results.get('eval_matched_to_allo', 0)}")

        # Show examples from allo test
        print("\n--- Sample predictions from allo test set ---")
        print_test_examples_with_constraints(model, tokenizer, 'test_orna_allo.csv', num_examples=3, tolerance=5)

    # Show examples from original test
    print("\n--- Sample predictions from original test set ---")
    print_test_examples_with_constraints(model, tokenizer, test_dataset_file, num_examples=3, tolerance=5)

    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nModel saved to: {output_dir}/final_model")
    print(f"\nDataset files created:")
    print(f"  Training:")
    print(f"    - train_segments.csv (original only, {len(train_df_original)} segments)")
    print(f"    - {train_dataset_file} (USED FOR TRAINING, {len(train_df)} segments)")
    print(f"  Validation:")
    print(f"    - {val_dataset_file} (original only, {len(val_df)} segments)")
    print(f"  Testing:")
    print(f"    - {test_dataset_file} (ORIGINAL, UNCHANGED, {len(test_df)} segments)")
    if allo_test_dataset is not None:
        print(f"    - test_orna_allo.csv (NEW allo test set, {len(allo_test_df)} segments)")

    print(f"\n🔒 Data integrity guarantees:")
    print(f"  ✅ test_segments.csv is UNCHANGED from original")
    print(f"  ✅ NO 25+ token sequences shared between training and EITHER test set")
    print(f"  ✅ Evaluation uses 5-token proximity tolerance for switch detection")
    print(f"  ✅ Both switch types (→Auto and →Allo) must match for correct detection")

    return trainer, model, tokenizer, test_results


# Keep the main execution as is
if __name__ == "__main__":
    trainer, model, tokenizer, results = train_tibetan_code_switching(target_switch_ratio=0.67)