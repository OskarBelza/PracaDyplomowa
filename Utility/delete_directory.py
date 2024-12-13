import os
import Config.config


def clear_directory(directory_path):
    """
    Remove all files and subdirectories from a specified directory.

    Args:
        directory_path (str): Path to the directory to clear.

    Returns:
        None
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


clear_directory(Config.config.FACE_PATH)