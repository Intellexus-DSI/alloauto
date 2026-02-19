"""
Diagnostic script to understand class distribution in your data
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def diagnose_data(token_csv='classify_allo_auto/data/tokens_3class.csv', sequence_csv='train_sequences.csv'):
    """Diagnose class imbalance in your dataset."""

    print("=== DATA DIAGNOSTIC REPORT ===\n")

    # Load token-level data
    if token_csv:
        try:
            token_df = pd.read_csv(token_csv)

            # Count classes
            class_counts = token_df['class_label'].value_counts().sort_index()
            total = len(token_df)

            print("TOKEN-LEVEL STATISTICS:")
            print(f"Total tokens: {total}")
            for cls in [0, 1, 2]:
                count = class_counts.get(cls, 0)
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  Class {cls}: {count} ({percentage:.2f}%)")

            # Calculate imbalance ratio
            if 1 in class_counts and 2 in class_counts:
                imbalance_ratio = class_counts[0] / min(class_counts[1], class_counts[2])
                print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")
                print("(How many class 0 tokens for each minority class token)")
        except Exception as e:
            print(f"Error loading token data: {e}")

    # Load sequence-level data
    if sequence_csv:
        try:
            seq_df = pd.read_csv(sequence_csv)

            print("\n\nSEQUENCE-LEVEL STATISTICS:")
            print(f"Total sequences: {len(seq_df)}")

            # Count labels in sequences
            all_labels = []
            for _, row in seq_df.iterrows():
                labels = list(map(int, row['labels'].split(',')))
                all_labels.extend(labels)

            label_counter = Counter(all_labels)

            print("\nLabel distribution across all sequences:")
            for cls in [0, 1, 2]:
                count = label_counter.get(cls, 0)
                percentage = (count / len(all_labels) * 100) if len(all_labels) > 0 else 0
                print(f"  Class {cls}: {count} ({percentage:.2f}%)")

            # Sequences with switches
            with_switches = seq_df['contains_switch'].sum()
            print(
                f"\nSequences containing switches: {with_switches} / {len(seq_df)} ({with_switches / len(seq_df) * 100:.1f}%)")

            # Average switches per sequence
            if 'num_switches' in seq_df.columns:
                avg_switches = seq_df['num_switches'].mean()
                max_switches = seq_df['num_switches'].max()
                print(f"Average switches per sequence: {avg_switches:.2f}")
                print(f"Max switches in a sequence: {max_switches}")
        except Exception as e:
            print(f"Error loading sequence data: {e}")

    print("\n\nRECOMMENDATIONS:")
    print("Your model is suffering from severe class imbalance.")
    print("With 0% F1 for classes 1 and 2, the model is just predicting class 0 for everything.")
    print("\nTo fix this, try:")
    print("1. Oversample minority classes by 20-50x")
    print("2. Use focal loss instead of cross-entropy")
    print("3. Set class weights: [1, 50, 50] or higher")
    print("4. Focus training on sequences with switches")
    print("5. Use a much lower learning rate (1e-5 or 5e-6)")
    print("6. Train for more epochs (20-30)")

    return token_df if token_csv else None, seq_df if sequence_csv else None


def visualize_distribution(sequence_csv='train_sequences.csv'):
    """Visualize the class distribution."""
    try:
        df = pd.read_csv(sequence_csv)

        # Count classes across all sequences
        class_counts = {0: 0, 1: 0, 2: 0}
        for _, row in df.iterrows():
            labels = list(map(int, row['labels'].split(',')))
            for label in labels:
                if label in class_counts:
                    class_counts[label] += 1

        # Plot
        plt.figure(figsize=(10, 6))

        # Bar plot
        plt.subplot(1, 2, 1)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        bars = plt.bar(classes, counts)
        bars[0].set_color('green')
        bars[1].set_color('orange')
        bars[2].set_color('red')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.yscale('log')  # Log scale to see minority classes

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{count}', ha='center', va='bottom')

        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=['No switch', 'To auto', 'To allo'],
                autopct='%1.1f%%', colors=['green', 'orange', 'red'])
        plt.title('Class Distribution (%)')

        plt.tight_layout()
        plt.savefig('class_distribution.png')
        print("\nVisualization saved to 'class_distribution.png'")
        plt.show()

    except Exception as e:
        print(f"Error creating visualization: {e}")


# Run diagnostics
if __name__ == "__main__":
    # Diagnose your data
    token_df, seq_df = diagnose_data('tokens_3class.csv', 'train_sequences.csv')

    # Visualize
    visualize_distribution('train_sequences.csv')

    # Quick check on a few sequences
    if seq_df is not None:
        print("\n\nSAMPLE SEQUENCES WITH SWITCHES:")
        switch_samples = seq_df[seq_df['contains_switch'] == True].head(3)
        for idx, row in switch_samples.iterrows():
            tokens = row['tokens'].split()[:20]  # First 20 tokens
            labels = list(map(int, row['labels'].split(',')))[:20]

            print(f"\nSequence {idx}:")
            for t, l in zip(tokens, labels):
                if l > 0:
                    print(f"  → {t} [CLASS {l}]")
                else:
                    print(f"    {t}")