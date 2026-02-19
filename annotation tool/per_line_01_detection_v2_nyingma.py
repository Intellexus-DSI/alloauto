#!/usr/bin/env python3
"""
Tibetan Verse Detection Script
Analyzes Classical Tibetan texts to identify verse sections based on syllable patterns.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

ignore_chars: List[str] = ["/", "@", "#", ";", ":", "(", ")"]
meter_counts: List[int] = [7, 9, 11]


def clean_line(line: str) -> str:
    """
    Clean a line by removing ignored characters.
    """
    for char in ignore_chars:
        line = line.replace(char, "")
    # replace all sequences of more than one space with a space
    line = re.sub(r"\s+", " ", line).strip()
    return line


def analyze_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    segment_sections = []
    buffer_lines = []   # collect consecutive short lines
    buffer_start = None # track first line number

    for i, line in enumerate(lines):
        line = clean_line(line)
        if not line:
            continue

        # add current line to buffer
        buffer_lines.append(line)
        if buffer_start is None:
            buffer_start = i

        # calculate total words across buffer
        total_words = sum(len(l.split()) for l in buffer_lines)

        if total_words >= 15:
            # save the buffered lines as a segment
            segment_sections.append(
                {
                    "text": os.path.basename(filepath),
                    "line_numbers": [buffer_start, i],
                    "lines": buffer_lines,  # store as list
                }
            )
            buffer_lines = []
            buffer_start = None

    # if something is left in buffer and never reached threshold
    if buffer_lines:
        segment_sections.append(
            {
                "text": os.path.basename(filepath),
                "final_line_number": buffer_start + len(buffer_lines) - 1,
                "lines": buffer_lines[:],
            }
        )

    return segment_sections




def main():
    """Main function to process all files in the corpus directory."""

    corpus_dir = "./Sungbum_flat_shortened_test"  # Change to your corpus directory
    output_dir = "./per_line_step_1_output"  # Change to your output directory

    if not os.path.exists(corpus_dir):
        print(f"Error: Directory {corpus_dir} not found!")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    processed_files = 0

    # Process all .txt files in the directory
    for filename in sorted(os.listdir(corpus_dir)):
            print(filename)
            filepath = os.path.join(corpus_dir, filename)
            print(f"Processing {filename}...")

            file_segments = analyze_file(filepath)

            # Save individual file results
            if filename.endswith(".txt"):
                output_filename = filename.replace(".txt", "_verse_analysis.json")
            elif filename.endswith(".TXT"):
                output_filename = filename.replace(".TXT", "_verse_analysis.json")

            output_filepath = os.path.join(output_dir, output_filename)

            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(file_segments, f, ensure_ascii=False, indent=2)

            if file_segments:
                print(f"  Found {len(file_segments)} segments -> {output_filename}")
            else:
                print(f"  No segments found -> {output_filename}")

            processed_files += 1

            # Progress update every 100 files
            if processed_files % 100 == 0:
                print(
                    f"Processed {processed_files} files")

    print("\nAnalysis complete!")
    print(f"Processed {processed_files} files")
    print(f"Individual results saved to {output_dir}/ directory")


if __name__ == "__main__":
    main()
