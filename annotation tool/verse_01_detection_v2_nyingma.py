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


def count_syllables(line: str) -> int:
    """
    Count syllables in a Tibetan line by removing ignored characters
    and splitting on whitespace.
    """
    # Remove ignored characters but keep periods and apostrophes
    cleaned = re.sub(r"[/@#;:()]", "", line.strip()).strip()
    if not cleaned:
        return 0
    return len(cleaned.split())


def calculate_verse_probability(
    syllable_counts: List[int], target_meters: List[int] = [7, 9, 11]
) -> Tuple[float, int]:
    """
    Calculate probability that a sequence of lines is verse.
    Returns (probability, best_matching_meter)
    """
    if not syllable_counts:
        return 0.0, 0

    best_prob = 0.0
    best_meter = 0

    for target in target_meters:
        total_weight = 0.0
        max_weight = 0.0

        for count in syllable_counts:
            if count == target:
                weight = 1.0  # Exact match = 100%
            elif abs(count - target) == 1:
                weight = 0.5  # ±1 syllable = 50%
            else:
                weight = 0.0  # No match = 0%
                total_weight = 0.0
                max_weight = 1.0
                break

            total_weight += weight
            max_weight += 1.0

        probability = total_weight / max_weight if max_weight > 0 else 0.0

        if probability > best_prob:
            best_prob = probability
            best_meter = target

    return best_prob, best_meter


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


def analyze_file(
    filepath: str,
    window_size: int = 4,
    min_section_length: int = 4,
    min_probability: float = 0.7,
    min_gap: int = 0,
) -> List[Dict]:
    """Analyze a single file for verse sections."""

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    # Clean and filter lines
    cleaned_lines = []
    for line in lines:
        line = clean_line(line)
        if line:
            cleaned_lines.append(line)

    if len(cleaned_lines) < min_section_length:
        return []

    # Calculate syllable counts for all lines
    syllable_counts = [count_syllables(line) for line in cleaned_lines]

    verse_sections = []
    i = 0

    while i <= len(cleaned_lines) - window_size:
        # get the meter of this line
        if syllable_counts[i] not in meter_counts:
            i += 1
            continue
        current_meter = syllable_counts[i]

        # now walk through the next lines until our probability of a verse section
        # drops below the minimum
        current_weight = 1.0
        current_max_weight = 1.0
        section_start = i
        section_end = i
        next_line = i + 1
        while (
            next_line < len(cleaned_lines)
            and (current_weight / current_max_weight) >= min_probability
        ):
            syllable_count = syllable_counts[next_line]
            if syllable_count == current_meter:
                current_weight += 1.0
            elif abs(syllable_count - current_meter) == 1:
                current_weight += 0.5
            else:
                break
            current_max_weight += 1.0
            section_end = next_line
            next_line += 1
        # If we found a valid section, check its length
        if section_end - section_start + 1 >= min_section_length:
            verse_sections.append(
                {
                    "text": os.path.basename(filepath),
                    "lines_included": [section_start, section_end],
                    "probability": round(current_weight / current_max_weight, 3),
                    "meter": current_meter,
                    "lines": cleaned_lines[section_start : section_end + 1],
                }
            )

        # Move to the next line after the current section
        i = section_end + 1

    return verse_sections


def main():
    """Main function to process all files in the corpus directory."""

    corpus_dir = "./Sungbum_flat_shortened"  # Change to your corpus directory
    output_dir = "./step1_output"  # Change to your output directory

    if not os.path.exists(corpus_dir):
        print(f"Error: Directory {corpus_dir} not found!")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    all_verse_sections = []
    processed_files = 0

    # Process all .txt files in the directory
    for filename in sorted(os.listdir(corpus_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(corpus_dir, filename)
            print(f"Processing {filename}...")

            sections = analyze_file(filepath)

            # Save individual file results
            output_filename = filename.replace(".txt", "_verse_analysis.json")
            output_filepath = os.path.join(output_dir, output_filename)

            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)

            all_verse_sections.extend(sections)

            if sections:
                print(f"  Found {len(sections)} verse sections -> {output_filename}")
            else:
                print(f"  No verse sections found -> {output_filename}")

            processed_files += 1

            # Progress update every 100 files
            if processed_files % 100 == 0:
                print(
                    f"Processed {processed_files} files, "
                    + f"found {len(all_verse_sections)} total sections"
                )

    print("\nAnalysis complete!")
    print(f"Processed {processed_files} files")
    print(f"Found {len(all_verse_sections)} verse sections")
    print(f"Individual results saved to {output_dir}/ directory")

    # Print some statistics
    if all_verse_sections:
        probs = [section["probability"] for section in all_verse_sections]
        meters = [section["meter"] for section in all_verse_sections]
        lengths = [
            section["lines_included"][1] - section["lines_included"][0] + 1
            for section in all_verse_sections
        ]

        print("\nStatistics:")
        print(f"Average probability: {sum(probs) / len(probs):.3f}")
        print(f"Meter distribution: {Counter(meters)}")
        print(f"Average section length: {sum(lengths) / len(lengths):.1f} lines")


if __name__ == "__main__":
    main()
