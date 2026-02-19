import os
import shutil

def flatten_directory(source_dir: str, dest_dir: str):
    """
    Flattens all files from a source directory's subfolders into a single
    destination directory.

    Args:
        source_dir (str): The path to the source directory containing subfolders.
        dest_dir (str): The path to the destination directory where all files
                        will be moved.
    """
    # Check if the source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Flattening files from '{source_dir}' to '{dest_dir}'...")

    # Walk through the source directory, including all subdirectories
    for dirpath, _, filenames in os.walk(source_dir):
        print (f"Current directory: {dirpath}")
        print (f"Filenames: {filenames}")
        # Skip the root source directory itself to avoid moving files from there
        if dirpath == source_dir:
            continue

        for filename in filenames:
            # Construct the full path to the source file
            source_file_path = os.path.join(dirpath, filename)

            # Construct the full path for the destination file
            dest_file_path = os.path.join(dest_dir, filename)

            # Handle potential filename collisions by appending a counter
            counter = 1
            original_filename, extension = os.path.splitext(filename)
            while os.path.exists(dest_file_path):
                new_filename = f"{original_filename}_{counter}{extension}"
                dest_file_path = os.path.join(dest_dir, new_filename)
                counter += 1

            try:
                # Move the file from the source to the destination
                shutil.move(source_file_path, dest_file_path)
                print(f"Moved: '{source_file_path}' -> '{dest_file_path}'")
                print("final counter:", counter)
            except Exception as e:
                print(f"Error moving file '{source_file_path}': {e}")

    print("\nFile flattening process complete.")
    print("Empty subdirectories might still remain in the source folder.")

if __name__ == "__main__":
    # --- Instructions for use ---
    # 1. Replace the paths below with your actual source and destination folders.
    # 2. Make sure the destination folder is not a subfolder of the source folder.
    # 3. This script moves files, it does not copy them.
    # ----------------------------

    # Example usage:
    # IMPORTANT: Change these paths to your own!
    source_folder = "./acip_sungbum_tshad_split_ewts"  # Change to your source folder
    destination_folder = "./Sungbum_flat"  # Change to your destination folder

    flatten_directory(source_folder, destination_folder)