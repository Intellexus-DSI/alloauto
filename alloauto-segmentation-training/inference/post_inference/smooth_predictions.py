import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

def get_simplified_label(pred_id: int) -> str:
    # 0: Non-switch Auto -> auto
    # 1: Non-switch Allo -> allo
    # 2: Switch->Auto -> auto
    # 3: Switch->Allo -> allo
    if pred_id in [0, 2]:
        return "auto"
    elif pred_id in [1, 3]:
        return "allo"
    else:
        return "unknown"

def identify_runs(words_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw word data into contiguous runs of the same simplified label.
    """
    if not words_data:
        return []

    runs = []
    current_label = get_simplified_label(words_data[0]['pred_id'])
    current_words = [words_data[0]['word']]
    start_idx = 0

    for i, item in enumerate(words_data[1:], start=1):
        item_label = get_simplified_label(item['pred_id'])
        if item_label == current_label:
            current_words.append(item['word'])
        else:
            runs.append({
                'type': current_label,
                'words': current_words,
                'start_idx': start_idx,
                'end_idx': i - 1, # Inclusive
                'length': len(current_words)
            })
            current_label = item_label
            current_words = [item['word']]
            start_idx = i
    
    # Append last run
    runs.append({
        'type': current_label,
        'words': current_words,
        'start_idx': start_idx,
        'end_idx': len(words_data) - 1,
        'length': len(current_words)
    })
    
    return runs

def consolidate_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge adjacent runs of the same type.
    """
    if not runs:
        return []
    new_runs = []
    curr = runs[0]
    for next_r in runs[1:]:
        if next_r['type'] == curr['type']:
            curr['words'].extend(next_r['words'])
            curr['length'] += next_r['length']
            curr['end_idx'] = next_r['end_idx']
        else:
            new_runs.append(curr)
            curr = next_r
    new_runs.append(curr)
    return new_runs

def smooth_simple(words_data: List[Dict[str, Any]], min_length: int) -> List[Dict[str, Any]]:
    """
    Smooth predictions using simple threshold merging.
    Any segment < min_length is merged into its neighbor (preferring previous).
    """
    if not words_data:
        return []

    # Initial runs
    runs = identify_runs(words_data)
    
    while True:
        # Find the first run that is too short
        short_idx = -1
        for idx, r in enumerate(runs):
            if r['length'] < min_length:
                short_idx = idx
                break
        
        if short_idx == -1:
            break # No short runs left
            
        if len(runs) == 1:
            # Only one run left, cannot merge
            break
            
        # Merge logic
        if short_idx > 0:
            # Merge into previous: Change type to match previous
            runs[short_idx]['type'] = runs[short_idx - 1]['type']
        else:
            # First run is short: Merge into next
            runs[short_idx]['type'] = runs[short_idx + 1]['type']
            
        # Consolidate adjacent same-type runs
        runs = consolidate_runs(runs)
        
    return [{'type': s['type'], 'text': ' '.join(s['words'])} for s in runs]

def main():
    parser = argparse.ArgumentParser(description="Smooth inference CSV predictions to JSONL segments.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--theta", type=int, default=10, help="Minimum token count for a valid segment. Default 10.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: File {csv_path} not found.")
        return

    # Read CSV
    words_data = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both old and new CSV column names
                pred_id_key = 'Decided_Prediction_ID' if 'Decided_Prediction_ID' in row else 'Prediction_ID'
                
                words_data.append({
                    'word': row['Word'],
                    'pred_id': int(row[pred_id_key])
                })
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not words_data:
        print("Error: Empty CSV or no data found.")
        return

    # Process
    smoothed_output = smooth_simple(words_data, args.theta)
    
    print(f"Smoothed segments count: {len(smoothed_output)}")

    # Output JSONL
    output_jsonl_path = csv_path.with_suffix('.jsonl')
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in smoothed_output:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Output CSV (smoothed)
    output_csv_path = csv_path.parent / f"{csv_path.stem}_smoothed.csv"
    try:
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['type', 'text'])
            writer.writeheader()
            writer.writerows(smoothed_output)
        print(f"Saved smoothed CSV results to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving smoothed CSV: {e}")

    print(f"Saved smoothed JSONL results to: {output_jsonl_path}")

if __name__ == "__main__":
    main()
