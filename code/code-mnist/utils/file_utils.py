
import os

def find_file_in_subdirectory(folder_path, file_name):
    print(folder_path)
    for root, dirs, files in os.walk(folder_path):
        if file_name in files:
            return os.path.join(root, file_name)

    # If the file is not found
    return None

def get_subfolder_paths(folder_path):
    subfolder_paths = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subfolder_paths

def get_subfolder_paths_level(main_folder, target_level):
    # List to store subfolders at the target level
    target_level_subfolders = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(main_folder):
        # Calculate the current level
        current_level = root[len(main_folder) + 1:].count(os.sep) + 1

        if current_level == target_level:
            # Found subfolders at the target level
            target_level_subfolders.extend(os.path.join(root, d) for d in dirs)

    # Print or use the list of subfolders at the target level as needed
    for subfolder in target_level_subfolders:
        print(f"Subfolder found: {subfolder}")

    return target_level_subfolders