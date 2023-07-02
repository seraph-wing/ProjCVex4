import os
import sys

def print_folder_structure(folder_path, output_file):
    with open(output_file, "w") as file:
        file.write(f"Folder Structure for: {folder_path}\n")
        file.write("\n")
        print_directory_contents(folder_path, file)

def print_directory_contents(path, file, indent_level=0):
    for item in os.listdir(path):
        if item.startswith('.'):  # Ignore hidden files and directories
            continue

        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            file.write(f"{'  ' * indent_level}+ {item}\n")
            print_directory_contents(item_path, file, indent_level + 1)
        elif not item.endswith('.jpg'):  # Ignore files with the extension .jpg
            file.write(f"{'  ' * indent_level}- {item}\n")

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 3:
    print("Usage: python folder_structure.py <folder_location> <output_file_name>")
else:
    # Get the folder location and output file name from the command-line arguments
    folder_location = sys.argv[1]
    output_file_name = sys.argv[2]
    output_file_path = os.path.join(folder_location, output_file_name)

    print_folder_structure(folder_location, output_file_path)
    print(f"Folder structure saved in '{output_file_path}'.")
