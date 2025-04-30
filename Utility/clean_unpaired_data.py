import os
import glob
from Config.config import SPECTROGRAM_PATH, FACE_PATH


def clean_unpaired_data(faces_dir, spectrograms_dir):
    """
    Remove unpaired .png files from the faces and spectrograms directories.
    Ensures that for each spectrogram file, a matching face image exists, and vice versa.

    Parameters:
        faces_dir (str): Path to the root directory containing face images, organized by emotion class.
        spectrograms_dir (str): Path to the root directory containing spectrogram images, organized by emotion class.

    Returns:
        None
    """
    # Get list of emotion subfolders (e.g. 'happy', 'angry', etc.)
    emotions = os.listdir(faces_dir)

    for emotion in emotions:
        print(f"🧹 Cleaning class: {emotion}")

        # Paths to face and spectrogram directories for the current emotion
        faces_path = os.path.join(faces_dir, emotion)
        specs_path = os.path.join(spectrograms_dir, emotion)

        # Collect sets of file names (without path) for faces and spectrograms
        face_files = set(os.path.basename(f) for f in glob.glob(os.path.join(faces_path, "*.png")))
        spec_files = set(os.path.basename(f) for f in glob.glob(os.path.join(specs_path, "*.png")))

        # Find common file names (paired files)
        common_files = face_files & spec_files

        # Determine which files are unpaired
        faces_to_remove = face_files - common_files
        specs_to_remove = spec_files - common_files

        # Delete unpaired face images
        for f in faces_to_remove:
            os.remove(os.path.join(faces_path, f))
            print(f"Removed face: {f}")

        # Delete unpaired spectrogram images
        for s in specs_to_remove:
            os.remove(os.path.join(specs_path, s))
            print(f"Removed spectrogram: {s}")

    print("✅ Dataset cleaned – only paired files remain.")


clean_unpaired_data(FACE_PATH, SPECTROGRAM_PATH)
