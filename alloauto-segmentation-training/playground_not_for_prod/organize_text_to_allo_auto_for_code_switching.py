import re
import pandas as pd
from docx import Document


def process_tibetan_docx(input_file, output_csv='code_switching_data.csv'):
    """
    Process a Tibetan .docx file with <allo> and <auto> tags
    and create a CSV for code-switching analysis.
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
        label = match.group(1)  # 'allo' or 'auto'
        segment_text = match.group(2).strip()

        if segment_text:
            # Handle Tibetan punctuation
            segment_text = segment_text.replace('//', ' // ')
            segment_text = segment_text.replace('/', ' / ')
            tokens = [t.strip() for t in segment_text.split() if t.strip()]

            for i, token in enumerate(tokens):
                # Determine if this is a switch point
                is_switch = (token_idx > 0 and i == 0 and label != previous_label)

                data.append({
                    'token_index': token_idx,
                    'token': token,
                    'label': label,
                    'is_switch_point': is_switch,
                    'switch_to': label if is_switch else None,
                    'iob_label': f"B-{label}" if (token_idx == 0 or is_switch) else f"I-{label}"
                })

                token_idx += 1

            previous_label = label

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8')

    # Print summary
    print(f"Processed {len(data)} tokens")
    print(f"Found {sum(1 for d in data if d['is_switch_point'])} switch points")
    print(f"Allo tokens: {sum(1 for d in data if d['label'] == 'allo')}")
    print(f"Auto tokens: {sum(1 for d in data if d['label'] == 'auto')}")
    print(f"\nData saved to: {output_csv}")
    print("\nFirst 10 rows:")
    print(df.head(10))

    return df


# Example usage:
if __name__ == "__main__":
    # Replace 'your_file.docx' with your actual file path
    df = process_tibetan_docx('classify_allo_auto/data/Nicola_Bajetta_rNam_gsum_bshad_pa_Auto_vs_Allo_signals_alo_and_auto_cleaned.docx', 'classify_allo_auto/data/tibetan_code_switching_annotated_allo_auto.csv')
