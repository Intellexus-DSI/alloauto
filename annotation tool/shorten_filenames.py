import os
import re
import shutil

def truncate_filename(filename: str, max_length: int = 150) -> str:
    """
    Sanitizes and truncates a filename to a safe length, removing invalid
    characters and ensuring it doesn't exceed a specified maximum length.

    Args:
        filename (str): The original filename.
        max_length (int): The maximum desired length for the filename,
                          including the file extension.

    Returns:
        str: A sanitized and truncated filename.
    """
    # Remove potentially invalid characters from the filename.
    # This pattern keeps letters, numbers, hyphens, underscores, and periods.
    sanitized = re.sub(r'[^\w\.\-]', '_', filename)

    # Get the file's name and extension.
    name, ext = os.path.splitext(sanitized)

    # Check if the filename is longer than the maximum allowed length.
    if len(sanitized) > max_length:
        # Calculate the available space for the name part after accounting for
        # the extension and a placeholder.
        available_name_length = max_length - len(ext) - 4 # 4 for the '...'

        if available_name_length <= 0:
            # If the extension is already too long, just truncate it.
            return sanitized[:max_length]

        # Truncate the name part and add a placeholder to indicate it was cut.
        truncated_name = name[:available_name_length]
        return f"{truncated_name}{ext}"

    return sanitized


def main():
    """
    Iterates through a source directory, shortens filenames, and copies
    the files to a new destination directory.
    """
    # Define the source and destination directories.
    source_dir = "./Sungbum_flat"
    output_dir = "./Sungbum_flat_shortened"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing files from: {source_dir}")
    print(f"Saving shortened files to: {output_dir}\n")

    # Walk through the source directory.
    for root, _, files in os.walk(source_dir):
        for filename in files:
            source_path = os.path.join(root, filename)
            
            # Shorten the filename
            shortened_filename = truncate_filename(filename)
            output_path = os.path.join(output_dir, shortened_filename)
            
            # Handle potential filename collisions by adding a counter
            counter = 1
            while os.path.exists(output_path):
                name, ext = os.path.splitext(shortened_filename)
                new_filename = f"{name}_{counter}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                counter += 1

            try:
                # Copy the file to the new location with the shortened filename
                shutil.copy2(source_path, output_path)
                print(f"Copied: '{filename}' -> '{shortened_filename}'")
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")
    
    print("\nFile shortening process complete.")


if __name__ == "__main__":
    main()