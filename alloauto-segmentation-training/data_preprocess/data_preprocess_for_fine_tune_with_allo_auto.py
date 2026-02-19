"""
Simple Add-on Script: Augment Training Data with ALLO/AUTO Files
=================================================================
This script ONLY adds additional files from ALLO and AUTO directories
to your existing preprocessed train/val sets.

It does NOT modify the original preprocessing - just adds more data.
"""

import os
import re
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import document processing
try:
    from docx import Document
except ImportError:
    print("Installing python-docx...")
    os.system("pip install python-docx --break-system-packages")
    from docx import Document

# Configuration
SEED = 42
random.seed(SEED)

# Directories
PREPROCESSED_DIR = 'dataset/preprocessed_clean'  # Your existing preprocessed data
ALLO_DIR = 'alloauto-segmentation-training/data/data_allo_orna/'
AUTO_DIR = 'alloauto-segmentation-training/data/data_auto_Orna/'
OUTPUT_DIR = 'dataset/preprocessed_augmented'  # New output directory

# Parameters for additional files
ADDITIONAL_MAX_TOKENS = 500  # Maximum tokens per segment for ALLO/AUTO files
ADDITIONAL_MIN_TOKENS = 50   # Minimum tokens per segment
OVERLAP_TOKENS = 50          # Overlap for longer files

# Split ratio for additional data
VAL_SPLIT = 0.15  # 15% of additional data goes to validation


class AdditionalDataProcessor:
    """Process additional ALLO and AUTO files."""

    def __init__(self):
        self.token_pattern = re.compile(r'\S+')

    def extract_text_from_file(self, filepath: str) -> str:
        """Extract text from either .txt or .docx file."""
        filepath = Path(filepath)

        if filepath.suffix == '.txt':
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"  Error reading {filepath.name}: {e}")
                return ""
        elif filepath.suffix == '.docx':
            try:
                doc = Document(filepath)
                return '\n'.join([para.text for para in doc.paragraphs])
            except Exception as e:
                print(f"  Error reading {filepath.name}: {e}")
                return ""
        else:
            return ""

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization for Tibetan text."""
        return self.token_pattern.findall(text)

    def create_labels(self, tokens: List[str], tag: str) -> List[int]:
        """Create token-level labels based on tag (auto=0, allo=1)."""
        if tag == 'auto':
            return [0] * len(tokens)
        else:  # allo
            return [1] * len(tokens)

    def process_file(self, filepath: str, tag: str) -> List[Dict]:
        """Process a single ALLO or AUTO file into segments."""
        text = self.extract_text_from_file(filepath)
        if not text:
            return []

        # Remove any potential tags (shouldn't be any in ALLO/AUTO files, but just in case)
        text = re.sub(r'<[^>]+>', '', text)

        tokens = self.tokenize(text)
        if not tokens:
            return []

        segments = []
        file_name = Path(filepath).name

        # If file is short, take it as one segment
        if len(tokens) <= ADDITIONAL_MAX_TOKENS:
            if len(tokens) >= ADDITIONAL_MIN_TOKENS:
                labels = self.create_labels(tokens, tag)
                segments.append({
                    'segment_id': f"{tag}_{file_name}_{0}",
                    'source_file': file_name,
                    'file_type': tag,
                    'tokens': ' '.join(tokens),
                    'labels': ','.join(map(str, labels)),
                    'num_tokens': len(tokens),
                    'num_transitions': 0,
                    'has_switch': False,
                    'original_text': ' '.join(tokens),
                    'is_additional': True  # Mark as additional data
                })
        else:
            # Split into segments with overlap
            for i, start_idx in enumerate(range(0, len(tokens), ADDITIONAL_MAX_TOKENS - OVERLAP_TOKENS)):
                end_idx = min(start_idx + ADDITIONAL_MAX_TOKENS, len(tokens))
                segment_tokens = tokens[start_idx:end_idx]

                if len(segment_tokens) >= ADDITIONAL_MIN_TOKENS:
                    labels = self.create_labels(segment_tokens, tag)
                    segments.append({
                        'segment_id': f"{tag}_{file_name}_{i}",
                        'source_file': file_name,
                        'file_type': tag,
                        'tokens': ' '.join(segment_tokens),
                        'labels': ','.join(map(str, labels)),
                        'num_tokens': len(segment_tokens),
                        'num_transitions': 0,
                        'has_switch': False,
                        'original_text': ' '.join(segment_tokens),
                        'is_additional': True
                    })

        return segments


def main():
    """Main function to augment existing data with ALLO/AUTO files."""

    print("="*80)
    print("AUGMENTING TRAINING DATA WITH ALLO/AUTO FILES")
    print("="*80)

    processor = AdditionalDataProcessor()

    # Step 1: Load existing preprocessed data
    print("\n" + "="*60)
    print("STEP 1: Loading existing preprocessed data")
    print("="*60)

    try:
        # Try to load the updated version first
        train_df = pd.read_csv(f'{PREPROCESSED_DIR}/train_segments_updated_v1.csv')
        val_df = pd.read_csv(f'{PREPROCESSED_DIR}/val_segments_updated_v1.csv')
        test_df = pd.read_csv(f'{PREPROCESSED_DIR}/test_segments_updated_v1.csv')
        print(f"✓ Loaded updated_v1 files")
    except:
        try:
            # Fall back to enhanced version
            train_df = pd.read_csv(f'{PREPROCESSED_DIR}/train_segments_enhanced.csv')
            val_df = pd.read_csv(f'{PREPROCESSED_DIR}/val_segments_enhanced.csv')
            test_df = pd.read_csv(f'{PREPROCESSED_DIR}/test_segments_original.csv')
            print(f"✓ Loaded enhanced files")
        except Exception as e:
            print(f"✗ Error loading preprocessed data: {e}")
            return

    print(f"  Original Train: {len(train_df)} segments")
    print(f"  Original Val: {len(val_df)} segments")
    print(f"  Original Test: {len(test_df)} segments (will not be modified)")

    # Step 2: Process ALLO files
    print("\n" + "="*60)
    print("STEP 2: Processing ALLO files")
    print("="*60)

    allo_segments = []
    if os.path.exists(ALLO_DIR):
        allo_files = list(Path(ALLO_DIR).glob('*.txt')) + list(Path(ALLO_DIR).glob('*.docx'))
        print(f"Found {len(allo_files)} ALLO files")

        for filepath in allo_files:
            print(f"  Processing: {filepath.name}")
            segments = processor.process_file(filepath, 'allo')
            if segments:
                allo_segments.extend(segments)
                print(f"    → {len(segments)} segments")
            else:
                print(f"    → No valid segments")
    else:
        print(f"✗ ALLO directory not found: {ALLO_DIR}")

    print(f"\nTotal ALLO segments: {len(allo_segments)}")

    # Step 3: Process AUTO files (only with 'citation')
    print("\n" + "="*60)
    print("STEP 3: Processing AUTO files (with 'citation' in name)")
    print("="*60)

    auto_segments = []
    if os.path.exists(AUTO_DIR):
        all_auto_files = list(Path(AUTO_DIR).glob('*.txt')) + list(Path(AUTO_DIR).glob('*.docx'))
        citation_files = [f for f in all_auto_files if 'citation' in f.name.lower()]

        print(f"Found {len(all_auto_files)} total AUTO files")
        print(f"Found {len(citation_files)} files with 'citation' in name")

        for filepath in citation_files:
            print(f"  Processing: {filepath.name}")
            segments = processor.process_file(filepath, 'auto')
            if segments:
                auto_segments.extend(segments)
                print(f"    → {len(segments)} segments")
            else:
                print(f"    → No valid segments")
    else:
        print(f"✗ AUTO directory not found: {AUTO_DIR}")

    print(f"\nTotal AUTO segments: {len(auto_segments)}")

    # Step 4: Balance ALLO and AUTO segments
    print("\n" + "="*60)
    print("STEP 4: Balancing ALLO and AUTO segments")
    print("="*60)

    n_allo = len(allo_segments)
    n_auto = len(auto_segments)

    if n_allo == 0 and n_auto == 0:
        print("✗ No additional segments found!")
        print("  Check that ALLO and AUTO directories exist and contain files")
        return

    if n_auto > n_allo and n_allo > 0:
        # Sample AUTO to match ALLO
        auto_segments = random.sample(auto_segments, n_allo)
        print(f"Sampled {n_allo} AUTO segments to match ALLO")
    elif n_allo > n_auto and n_auto > 0:
        # Sample ALLO to match AUTO
        allo_segments = random.sample(allo_segments, n_auto)
        print(f"Sampled {n_auto} ALLO segments to match AUTO")

    print(f"Final counts: {len(allo_segments)} ALLO, {len(auto_segments)} AUTO")

    # Step 5: Split additional segments into train/val
    print("\n" + "="*60)
    print("STEP 5: Splitting additional segments")
    print("="*60)

    additional_segments = allo_segments + auto_segments
    random.shuffle(additional_segments)

    n_additional = len(additional_segments)
    n_additional_val = int(n_additional * VAL_SPLIT)
    n_additional_train = n_additional - n_additional_val

    additional_val = additional_segments[:n_additional_val]
    additional_train = additional_segments[n_additional_val:]

    print(f"Additional segments: {n_additional} total")
    print(f"  → {n_additional_train} to train")
    print(f"  → {n_additional_val} to validation")

    # Step 6: Combine with original data
    print("\n" + "="*60)
    print("STEP 6: Combining with original data")
    print("="*60)

    # Add is_additional column to original data if not present
    if 'is_additional' not in train_df.columns:
        train_df['is_additional'] = False
    if 'is_additional' not in val_df.columns:
        val_df['is_additional'] = False

    # Convert additional segments to DataFrames
    if additional_train:
        additional_train_df = pd.DataFrame(additional_train)
        augmented_train_df = pd.concat([train_df, additional_train_df], ignore_index=True)
    else:
        augmented_train_df = train_df

    if additional_val:
        additional_val_df = pd.DataFrame(additional_val)
        augmented_val_df = pd.concat([val_df, additional_val_df], ignore_index=True)
    else:
        augmented_val_df = val_df

    # Shuffle the augmented sets
    augmented_train_df = augmented_train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    augmented_val_df = augmented_val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Augmented Train: {len(augmented_train_df)} segments")
    print(f"  - Original: {(~augmented_train_df['is_additional']).sum()}")
    print(f"  - Additional: {augmented_train_df['is_additional'].sum()}")

    print(f"Augmented Val: {len(augmented_val_df)} segments")
    print(f"  - Original: {(~augmented_val_df['is_additional']).sum()}")
    print(f"  - Additional: {augmented_val_df['is_additional'].sum()}")

    print(f"Test: {len(test_df)} segments (unchanged)")

    # Step 7: Save augmented datasets
    print("\n" + "="*60)
    print("STEP 7: Saving augmented datasets")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    augmented_train_df.to_csv(f'{OUTPUT_DIR}/train_segments_with_more_auto_allo.csv', index=False)
    augmented_val_df.to_csv(f'{OUTPUT_DIR}/val_segments_more_auto_allo.csv', index=False)
    test_df.to_csv(f'{OUTPUT_DIR}/test_segments_original.csv', index=False)  # Copy test unchanged

    print(f"Saved to {OUTPUT_DIR}/")
    print(f"  - train_segments_with_more_auto_allo.csv")
    print(f"  - val_segments_more_auto_allo.csv")
    print(f"  - test_segments_original.csv")

    # Step 8: Print final statistics
    print("\n" + "="*80)
    print("AUGMENTATION COMPLETE - STATISTICS")
    print("="*80)

    def print_stats(df, name):
        """Print statistics for a dataset."""
        print(f"\n{name} Statistics:")
        print(f"  Total segments: {len(df)}")

        # Token stats
        print(f"  Total tokens: {df['num_tokens'].sum()}")
        print(f"  Avg tokens/segment: {df['num_tokens'].mean():.1f}")
        print(f"  Min tokens: {df['num_tokens'].min()}")
        print(f"  Max tokens: {df['num_tokens'].max()}")

        # Source stats
        if 'is_additional' in df.columns:
            original = (~df['is_additional']).sum()
            additional = df['is_additional'].sum()
            print(f"  Original segments: {original} ({original/len(df)*100:.1f}%)")
            print(f"  Additional segments: {additional} ({additional/len(df)*100:.1f}%)")

        # File type distribution
        if 'file_type' in df.columns:
            type_counts = df['file_type'].value_counts()
            print(f"  File types:")
            for ftype, count in type_counts.items():
                print(f"    - {ftype}: {count} ({count/len(df)*100:.1f}%)")

        # Label distribution
        all_labels = []
        for labels_str in df['labels']:
            all_labels.extend([int(l) for l in labels_str.split(',')])

        label_counts = {i: all_labels.count(i) for i in range(4)}
        total_labels = sum(label_counts.values())

        print(f"  Label distribution:")
        print(f"    - Auto (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total_labels*100:.1f}%)")
        print(f"    - Allo (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total_labels*100:.1f}%)")
        print(f"    - Switch→Auto (2): {label_counts.get(2, 0)} ({label_counts.get(2, 0)/total_labels*100:.1f}%)")
        print(f"    - Switch→Allo (3): {label_counts.get(3, 0)} ({label_counts.get(3, 0)/total_labels*100:.1f}%)")

        # Segments with switches
        if 'has_switch' in df.columns:
            with_switches = df['has_switch'].sum()
            print(f"  Segments with switches: {with_switches} ({with_switches/len(df)*100:.1f}%)")

    print_stats(augmented_train_df, "AUGMENTED TRAIN")
    print_stats(augmented_val_df, "AUGMENTED VALIDATION")
    print_stats(test_df, "TEST (UNCHANGED)")

    print("\n" + "="*80)
    print("✓ Augmentation complete!")
    print("  Your original preprocessed data has been augmented with ALLO/AUTO files")
    print("  The augmented datasets are ready for training")
    print("="*80)


if __name__ == "__main__":
    main()