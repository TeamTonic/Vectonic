from typing import List
import os

def get_all_files(directory: str) -> List[str]:
    """
    Recursively get all file names from a directory.

    Args:
        directory: The directory path.

    Returns:
        A list of file names.
    """
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names