import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_ner_data(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split NER data while keeping segments together"""

    df = pd.read_csv(csv_path)

    # Find segment boundaries (where B- tags appear)
    segment_starts = []
    current_segment = []

    for idx, row in df.iterrows():
        if row['label'].startswith('B-') and current_segment:
            segment_starts.append(current_segment[0])
            current_segment = [idx]
        else:
            current_segment.append(idx)

    # Add last segment
    if current_segment:
        segment_starts.append(current_segment[0])

    # Create segments list
    segments = []
    for i in range(len(segment_starts) - 1):
        segments.append((segment_starts[i], segment_starts[i + 1]))
    segments.append((segment_starts[-1], len(df)))

    # Shuffle segments (not individual tokens!)
    np.random.shuffle(segments)

    # Split segments
    n_segments = len(segments)
    train_end = int(n_segments * train_ratio)
    val_end = train_end + int(n_segments * val_ratio)

    train_segments = segments[:train_end]
    val_segments = segments[train_end:val_end]
    test_segments = segments[val_end:]

    # Create dataframes
    def segments_to_df(segs):
        indices = []
        for start, end in segs:
            indices.extend(range(start, end))
        return df.iloc[indices].reset_index(drop=True)

    train_df = segments_to_df(train_segments)
    val_df = segments_to_df(val_segments)
    test_df = segments_to_df(test_segments)

    # Save splits
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    print(f"Data split complete:")
    print(f"Train: {len(train_df)} tokens ({len(train_segments)} segments)")
    print(f"Val: {len(val_df)} tokens ({len(val_segments)} segments)")
    print(f"Test: {len(test_df)} tokens ({len(test_segments)} segments)")

    return train_df, val_df, test_df


# Best options for Tibetan:
model_name = "bert-base-multilingual-cased"  # Supports 104 languages including Tibetan
# OR
model_name_2 = "xlm-roberta-base"  # Even better multilingual support

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class TibetanNERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = self.process_dataframe(dataframe)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = {
            'O': 0,
            'B-AUTO': 1,
            'I-AUTO': 2,
            'B-ALLO': 3,
            'I-ALLO': 4
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def process_dataframe(self, df):
        """Group tokens into sequences"""
        sequences = []
        current_seq = {'tokens': [], 'labels': []}

        for _, row in df.iterrows():
            current_seq['tokens'].append(row['token'])
            current_seq['labels'].append(row['label'])

            # Split into manageable sequences
            if len(current_seq['tokens']) >= 100:  # Adjust based on your needs
                sequences.append(current_seq)
                current_seq = {'tokens': [], 'labels': []}

        if current_seq['tokens']:
            sequences.append(current_seq)

        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        labels = item['labels']

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Align labels with tokenized input
        word_ids = encoding.word_ids()
        label_ids = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(self.label_to_id[labels[word_idx]])
            else:
                label_ids.append(-100)  # Subword tokens
            previous_word_idx = word_idx

        encoding['labels'] = torch.tensor(label_ids)

        return {key: val.squeeze() for key, val in encoding.items()}