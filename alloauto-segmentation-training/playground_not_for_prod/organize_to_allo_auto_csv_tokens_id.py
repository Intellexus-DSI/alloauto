import re
import csv
from docx import Document


def convert_to_ner_with_token_id(docx_path, output_csv, chunk_size=10000):
    """Convert to NER format with token_id for reference"""

    doc = Document(docx_path)

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Essential headers only
        writer.writerow(['token_id', 'token', 'label'])

        current_label = 'O'
        first_token = True
        token_id = 0

        for para in doc.paragraphs:
            if not para.text.strip():
                continue

            parts = re.split(r'(<auto>|<allo>)', para.text)

            for part in parts:
                if part == '<auto>':
                    current_label = 'AUTO'
                    first_token = True
                elif part == '<allo>':
                    current_label = 'ALLO'
                    first_token = True
                elif part.strip():
                    tokens = part.strip().split()

                    for token in tokens:
                        if current_label != 'O':
                            if first_token:
                                label = f'B-{current_label}'
                                first_token = False
                            else:
                                label = f'I-{current_label}'
                        else:
                            label = 'O'

                        writer.writerow([token_id, token, label])
                        token_id += 1

                        # Progress indicator
                        if token_id % chunk_size == 0:
                            print(f"Processed {token_id} tokens...")

    print(f"Total tokens processed: {token_id}")
    return token_id


def analyze_with_token_ids(csv_path):
    """Show why token_id is useful"""
    import pandas as pd

    df = pd.read_csv(csv_path)

    # 1. Get specific tokens
    print("Tokens 50-55:")
    print(df.iloc[50:55])

    # 2. Find transitions
    print("\nStyle transitions:")
    for i in range(1, len(df)):
        if df.iloc[i]['label'][0] == 'B':  # New segment starts
            print(
                f"Token {df.iloc[i]['token_id']}: {df.iloc[i - 1]['token']} -> {df.iloc[i]['token']} (new {df.iloc[i]['label']})")

    # 3. Extract specific segments
    print("\nExtract AUTO segment starting at token 100:")
    start_idx = 100
    segment_tokens = []
    for i in range(start_idx, len(df)):
        if df.iloc[i]['label'].endswith('AUTO'):
            segment_tokens.append(df.iloc[i]['token'])
        else:
            break
    print(' '.join(segment_tokens[:20]) + '...')


def main():
    # Convert your document
    docx_path = 'classify_allo_auto/data/Nicola_Bajetta_rNam_gsum_bshad_pa_Auto_vs_Allo_signals_alo_and_auto_cleaned.docx'
    output_csv = 'classify_allo_auto/data/tibetan_ner_tokens_allo_auto.csv'


    # Process
    total = convert_to_ner_with_token_id(docx_path, output_csv)

    # Now you can reference any token by ID
    import pandas as pd
    df = pd.read_csv(output_csv)

    # Example: "I want to see what's around token 500"
    context = df.iloc[495:505]
    print(f"\nContext around token 500:")
    print(context)

    # Example: "Show me all AUTO segments"
    auto_starts = df[df['label'] == 'B-AUTO']['token_id'].tolist()
    print(f"\nAUTO segments start at tokens: {auto_starts[:10]}...")


if __name__ == "__main__":
    main()