import os
import Config.config


def count_images(specs_dir):
    """
    Count all images in the 'Faces' directory, including subdirectories.

    Args:
        faces_dir (str): Path to the 'Faces' directory.

    Returns:
        int: Total number of image files in the directory and its subdirectories.
    """
    total_images = 0

    for root, _, files in os.walk(specs_dir):
        image_files = [file for file in files if file.lower().endswith('.npy')]
        total_images += len(image_files)

    print(f"Total number of images in '{specs_dir}': {total_images}")
    return total_images


# Example usage
spec_dir = Config.config.SPECTROGRAM_PATH
count_images(spec_dir)
