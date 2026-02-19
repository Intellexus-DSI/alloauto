import re
import pandas as pd
from docx import Document
import numpy as np


def process_tibetan_3class(input_file, output_csv='classify_allo_auto/data//cs_3class_data.csv',
                           sequences_csv='cs_sequences.csv',
                           sequence_length=512):
    """
    Process Tibetan text with 3-class labeling:
    0 = regular token (no switch)
    1 = switch to 'auto'
    2 = switch to 'allo'
    """
    # Read the document
    doc = Document(input_file)
    full_text = ' '.join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

    # Pattern to find tagged segments
    pattern = r'<(allo|auto)>(.*?)(?=<(?:allo|auto)>|$)'
    matches = list(re.finditer(pattern, full_text, re.DOTALL))

    # Process tokens
    data = []
    token_idx = 0
    previous_label = None

    for match in matches:
        current_label = match.group(1)  # 'allo' or 'auto'
        segment_text = match.group(2).strip()

        if segment_text:
            # Handle Tibetan punctuation
            segment_text = segment_text.replace('//', ' // ')
            segment_text = segment_text.replace('/', ' / ')
            tokens = [t.strip() for t in segment_text.split() if t.strip()]

            for i, token in enumerate(tokens):
                # Determine the class label
                if token_idx == 0:
                    # First token is not a switch
                    class_label = 0
                elif i == 0 and previous_label != current_label:
                    # This is a switch point
                    if current_label == 'auto':
                        class_label = 1  # Switch to auto
                    else:  # current_label == 'allo'
                        class_label = 2  # Switch to allo
                else:
                    # Regular token, no switch
                    class_label = 0

                data.append({
                    'token_index': token_idx,
                    'token': token,
                    'current_segment': current_label,
                    'class_label': class_label,
                    'is_switch': class_label > 0,
                    'switch_type': 'to_auto' if class_label == 1 else ('to_allo' if class_label == 2 else 'none')
                })

                token_idx += 1

            previous_label = current_label

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8')

    # Create sequences for BERT training
    sequences_data = create_bert_sequences(data, sequence_length)
    seq_df = pd.DataFrame(sequences_data)
    seq_df.to_csv(sequences_csv, index=False, encoding='utf-8')

    # Print summary
    print("=== Processing Summary ===")
    print(f"Total tokens: {len(data)}")
    print(f"Class 0 (no switch): {sum(1 for d in data if d['class_label'] == 0)}")
    print(f"Class 1 (switch to auto): {sum(1 for d in data if d['class_label'] == 1)}")
    print(f"Class 2 (switch to allo): {sum(1 for d in data if d['class_label'] == 2)}")
    print(f"\nToken-level data saved to: {output_csv}")
    print(f"Sequence data saved to: {sequences_csv}")
    print("\nFirst 20 tokens:")
    print(df.head(20))

    return df, seq_df


def create_bert_sequences(token_data, max_length=512, stride=64):
    """
    Create overlapping sequences for BERT training.
    Uses sliding window approach to capture context around switch points.
    """
    sequences = []

    # Create sliding windows
    for start_idx in range(0, len(token_data), stride):
        end_idx = min(start_idx + max_length, len(token_data))
        window = token_data[start_idx:end_idx]

        if len(window) > 10:  # Skip very short sequences
            tokens = [t['token'] for t in window]
            labels = [t['class_label'] for t in window]

            # Check if this sequence contains any switches
            has_switch = any(label > 0 for label in labels)

            sequences.append({
                'sequence_id': len(sequences),
                'tokens': ' '.join(tokens),
                'labels': ','.join(map(str, labels)),
                'sequence_length': len(tokens),
                'contains_switch': has_switch,
                'num_switches': sum(1 for label in labels if label > 0),
                'start_token_idx': start_idx,
                'end_token_idx': end_idx - 1
            })

    return sequences


def prepare_for_bert_training(sequences_csv, train_ratio=0.8):
    """
    Prepare data for BERT fine-tuning with train/validation split.
    """
    df = pd.read_csv(sequences_csv)

    # Prioritize sequences with switches for training
    switch_sequences = df[df['contains_switch'] == True]
    no_switch_sequences = df[df['contains_switch'] == False]

    # Shuffle
    switch_sequences = switch_sequences.sample(frac=1, random_state=42)
    no_switch_sequences = no_switch_sequences.sample(frac=1, random_state=42)

    # Split
    n_switch_train = int(len(switch_sequences) * train_ratio)
    n_no_switch_train = int(len(no_switch_sequences) * train_ratio)

    train_data = pd.concat([
        switch_sequences[:n_switch_train],
        no_switch_sequences[:n_no_switch_train]
    ]).sample(frac=1, random_state=42)

    val_data = pd.concat([
        switch_sequences[n_switch_train:],
        no_switch_sequences[n_no_switch_train:]
    ]).sample(frac=1, random_state=42)

    # Save splits
    train_data.to_csv('train_sequences.csv', index=False)
    val_data.to_csv('val_sequences.csv', index=False)

    print(f"\n=== Train/Val Split ===")
    print(f"Training sequences: {len(train_data)} ({train_data['contains_switch'].sum()} with switches)")
    print(f"Validation sequences: {len(val_data)} ({val_data['contains_switch'].sum()} with switches)")

    return train_data, val_data


# Additional utility function for creating balanced batches
def create_balanced_dataset(sequences_csv, output_csv='balanced_sequences.csv'):
    """
    Create a balanced dataset with equal representation of all classes.
    """
    df = pd.read_csv(sequences_csv)

    balanced_sequences = []

    for _, row in df.iterrows():
        labels = list(map(int, row['labels'].split(',')))
        tokens = row['tokens'].split()

        # Find all switch points in this sequence
        for i, label in enumerate(labels):
            if label > 0:  # Found a switch point
                # Create a focused sequence around this switch
                start = max(0, i - 20)
                end = min(len(tokens), i + 20)

                focused_tokens = tokens[start:end]
                focused_labels = labels[start:end]

                balanced_sequences.append({
                    'sequence_id': len(balanced_sequences),
                    'tokens': ' '.join(focused_tokens),
                    'labels': ','.join(map(str, focused_labels)),
                    'switch_position': i - start,  # Position of switch in the focused sequence
                    'switch_class': label,
                    'sequence_length': len(focused_tokens)
                })

    # Also add some sequences without switches
    no_switch_samples = df[df['contains_switch'] == False].sample(
        n=min(len(df[df['contains_switch'] == False]), len(balanced_sequences) // 2),
        random_state=42
    )

    for _, row in no_switch_samples.iterrows():
        balanced_sequences.append({
            'sequence_id': len(balanced_sequences),
            'tokens': row['tokens'],
            'labels': row['labels'],
            'switch_position': -1,
            'switch_class': 0,
            'sequence_length': row['sequence_length']
        })

    balanced_df = pd.DataFrame(balanced_sequences)
    balanced_df = balanced_df.sample(frac=1, random_state=42)  # Shuffle
    balanced_df.to_csv(output_csv, index=False)

    print(f"\n=== Balanced Dataset ===")
    print(f"Total sequences: {len(balanced_df)}")
    print(f"Class 0 (no switch): {len(balanced_df[balanced_df['switch_class'] == 0])}")
    print(f"Class 1 (to auto): {len(balanced_df[balanced_df['switch_class'] == 1])}")
    print(f"Class 2 (to allo): {len(balanced_df[balanced_df['switch_class'] == 2])}")

    return balanced_df


# Example usage
if __name__ == "__main__":
    # Process the document
    token_df, seq_df = process_tibetan_3class(
        'classify_allo_auto/data/Nicola_Bajetta_rNam_gsum_bshad_pa_Auto_vs_Allo_signals_alo_and_auto_cleaned.docx',
        'classify_allo_auto/data/tokens_3class_allo_auto.csv',
        'classify_allo_auto/data/sequences_3class_allo_auto.csv',
        sequence_length=512
    )

    # Prepare train/val split
    train_df, val_df = prepare_for_bert_training('classify_allo_auto/data/sequences_3class_allo_auto.csv')

    # Create balanced dataset for better training
    balanced_df = create_balanced_dataset('classify_allo_auto/data/sequences_3class_allo_auto.csv')
    import ipdb
    ipdb.set_trace()