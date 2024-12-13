import os
import Config.config


def count_images(faces_dir):
    """
    Count all images in the 'Faces' directory, including subdirectories.

    Args:
        faces_dir (str): Path to the 'Faces' directory.

    Returns:
        int: Total number of image files in the directory and its subdirectories.
    """
    total_images = 0

    for root, _, files in os.walk(faces_dir):
        image_files = [file for file in files if file.lower().endswith('.jpg')]
        total_images += len(image_files)

    print(f"Total number of images in '{faces_dir}': {total_images}")
    return total_images


# Example usage
faces_dir = Config.config.FACE_PATH
count_images(faces_dir)
