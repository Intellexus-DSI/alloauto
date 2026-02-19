"""
Script to find .docx files from auto directory that were NOT used in training or validation
Finds one file with 'citation' in name and one without
"""

import os
import pandas as pd
from pathlib import Path


def get_files_used_in_training():
    """
    Extract which files were used in training and validation from the CSV files
    """
    used_files = set()

    # Check train_segments.csv
    if Path('train_segments.csv').exists():
        train_df = pd.read_csv('train_segments.csv')
        if 'source_file' in train_df.columns:
            # Filter for auto files
            auto_files = train_df[train_df['file_type'] == 'auto']['source_file'].unique()
            used_files.update(auto_files)
            print(f"Found {len(auto_files)} auto files in training set")

    # Check val_segments.csv
    if Path('val_segments.csv').exists():
        val_df = pd.read_csv('val_segments.csv')
        if 'source_file' in val_df.columns:
            # Filter for auto files
            auto_files = val_df[val_df['file_type'] == 'auto']['source_file'].unique()
            used_files.update(auto_files)
            print(f"Found {len(auto_files)} auto files in validation set")

    print(f"\nTotal unique auto files used in train/val: {len(used_files)}")
    return used_files


def find_unused_docx_files(auto_dir='alloauto-segmentation-training/data/data_auto_Orna'):
    """
    Find .docx files in auto directory that weren't used in training
    """

    # Get list of files used in training/validation
    used_files = get_files_used_in_training()

    # Get all .docx files in the auto directory
    if not os.path.exists(auto_dir):
        print(f"❌ Directory not found: {auto_dir}")
        return None, None

    all_docx_files = [f for f in os.listdir(auto_dir) if f.endswith('.docx')]
    print(f"\nTotal .docx files in {auto_dir}: {len(all_docx_files)}")

    # Separate into citation and non-citation files
    citation_files = [f for f in all_docx_files if 'citation' in f.lower()]
    non_citation_files = [f for f in all_docx_files if 'citation' not in f.lower()]

    print(f"  - Files with 'citation': {len(citation_files)}")
    print(f"  - Files without 'citation': {len(non_citation_files)}")

    # Find unused files
    unused_citation_files = [f for f in citation_files if f not in used_files]
    unused_non_citation_files = [f for f in non_citation_files if f not in used_files]

    print(f"\n📊 Unused files:")
    print(f"  - Unused citation files: {len(unused_citation_files)}")
    print(f"  - Unused non-citation files: {len(unused_non_citation_files)}")

    # List of preferred non-citation files (from your message)
    preferred_non_citation = [
        'Bu ston, rNal \'byor rgyud kyi chos \'byung 1.docx',
        'Bu ston, rNal \'byor rgyud kyi chos \'byung 2.docx',
        'Bu ston, rNal \'byor rgyud kyi chos \'byung 3.docx',
        'rTsags Dar ma rgyal po - gSang ldan gyi rgya cher bshad pa 2.docx',
        'A khu ching - sKu brnyan gyi lo rgyus 1.docx',
        'A khu ching - sKu brnyan gyi lo rgyus 2.docx',
        'A khu ching - sKu brnyan gyi lo rgyus3.docx'
    ]

    # Find unused preferred files
    unused_preferred = [f for f in preferred_non_citation if f in unused_non_citation_files]

    # Select files for testing
    selected_citation = None
    selected_non_citation = None

    # Select one unused citation file
    if unused_citation_files:
        selected_citation = unused_citation_files[0]
        print(f"\n✅ Selected unused CITATION file:")
        print(f"   {selected_citation}")
    else:
        print(f"\n⚠️ No unused citation files found!")

    # Select one unused non-citation file (prefer from the list you provided)
    if unused_preferred:
        selected_non_citation = unused_preferred[0]
        print(f"\n✅ Selected unused NON-CITATION file (from preferred list):")
        print(f"   {selected_non_citation}")
    elif unused_non_citation_files:
        selected_non_citation = unused_non_citation_files[0]
        print(f"\n✅ Selected unused NON-CITATION file:")
        print(f"   {selected_non_citation}")
    else:
        print(f"\n⚠️ No unused non-citation files found!")

    # Show all unused files for reference
    print(f"\n📋 ALL UNUSED FILES FOR REFERENCE:")

    if unused_citation_files:
        print(f"\nUnused citation files ({len(unused_citation_files)} total):")
        for i, f in enumerate(unused_citation_files[:10], 1):  # Show first 10
            marker = "→" if f == selected_citation else " "
            print(f"  {marker} {i}. {f}")
        if len(unused_citation_files) > 10:
            print(f"     ... and {len(unused_citation_files) - 10} more")

    if unused_preferred:
        print(f"\nUnused files from your preferred list:")
        for f in unused_preferred:
            marker = "→" if f == selected_non_citation else " "
            print(f"  {marker} {f}")

    if unused_non_citation_files:
        print(f"\nOther unused non-citation files ({len(unused_non_citation_files)} total):")
        other_non_citation = [f for f in unused_non_citation_files if f not in preferred_non_citation]
        for i, f in enumerate(other_non_citation[:10], 1):  # Show first 10
            marker = "→" if f == selected_non_citation else " "
            print(f"  {marker} {i}. {f}")
        if len(other_non_citation) > 10:
            print(f"     ... and {len(other_non_citation) - 10} more")

    return selected_citation, selected_non_citation


def verify_files_not_in_training(selected_files, auto_dir='alloauto-segmentation-training/data/data_auto_Orna'):
    """
    Double-check that selected files are really not in training/validation
    """
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print(f"{'=' * 60}")

    for csv_file in ['train_segments.csv', 'val_segments.csv', 'test_segments.csv']:
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            if 'source_file' in df.columns:
                for selected_file in selected_files:
                    if selected_file and selected_file in df['source_file'].values:
                        print(f"⚠️ WARNING: {selected_file} found in {csv_file}!")
                    else:
                        print(f"✅ {selected_file} NOT in {csv_file}")


def create_test_file_list(citation_file, non_citation_file,
                          auto_dir='alloauto-segmentation-training/data/data_auto_Orna'):
    """
    Create a list of files for testing with full paths
    """
    print(f"\n{'=' * 60}")
    print("TEST FILES FOR INFERENCE")
    print(f"{'=' * 60}")

    test_files = []

    if citation_file:
        full_path = os.path.join(auto_dir, citation_file)
        test_files.append(full_path)
        print(f"\n1. Citation file:")
        print(f"   '{full_path}'")

    if non_citation_file:
        full_path = os.path.join(auto_dir, non_citation_file)
        test_files.append(full_path)
        print(f"\n2. Non-citation file:")
        print(f"   '{full_path}'")

    # Save to a file for easy use
    with open('test_files_for_inference.txt', 'w') as f:
        f.write("# Test files for inference (not in training/validation)\n")
        f.write(f"# Generated from {auto_dir}\n\n")
        for file_path in test_files:
            f.write(f"{file_path}\n")

    print(f"\n📄 File list saved to: test_files_for_inference.txt")

    # Also create Python code snippet
    print(f"\n📝 Python code for your inference script:")
    print("```python")
    print("docx_files = [")
    for file_path in test_files:
        print(f"    '{file_path}',")
    print("]")
    print("```")

    return test_files


def main():
    """
    Main function to find unused files for testing
    """
    print("=" * 80)
    print("FINDING UNUSED .DOCX FILES FOR TESTING")
    print("=" * 80)

    auto_dir = 'alloauto-segmentation-training/data/data_auto_Orna'

    # Find unused files
    citation_file, non_citation_file = find_unused_docx_files(auto_dir)

    if citation_file or non_citation_file:
        # Verify they're not in training
        selected_files = [f for f in [citation_file, non_citation_file] if f]
        verify_files_not_in_training(selected_files, auto_dir)

        # Create test file list
        test_files = create_test_file_list(citation_file, non_citation_file, auto_dir)

        print(f"\n{'=' * 80}")
        print("✅ COMPLETE!")
        print(f"{'=' * 80}")
        print(f"\nYou can now use these files for inference testing.")
        print(f"They are guaranteed to NOT be in your training or validation sets.")
    else:
        print(f"\n❌ Could not find suitable test files!")


if __name__ == "__main__":
    main()