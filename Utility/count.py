import os
from Config.config import FACE_PATH, SPECTROGRAM_PATH


def count_files(directory):
    """
    Count the number of .png or .npy files in each subdirectory (representing emotions) within a given directory.

    Parameters:
        directory (str): The path to the root directory containing subdirectories for each emotion.

    Returns:
        dict: Dictionary mapping emotion names to the number of files found.
    """
    emotion_counts = {}

    # Check if directory exists
    if os.path.exists(directory):
        for emotion in os.listdir(directory):
            emotion_path = os.path.join(directory, emotion)

            # Only process subdirectories
            if os.path.isdir(emotion_path):
                # Count only PNG or NPY files
                num_files = len([
                    f for f in os.listdir(emotion_path)
                    if f.endswith('.png') or f.endswith('.npy')
                ])
                emotion_counts[emotion] = num_files

    return emotion_counts


def count_images_and_spectrograms(face_dir=FACE_PATH, spec_dir=SPECTROGRAM_PATH):
    """
    Count face images and spectrograms per emotion and summarize the totals.

    Parameters:
        face_dir (str): Path to directory containing face images organized by emotion.
        spec_dir (str): Path to directory containing spectrograms organized by emotion.

    Returns:
        dict: Dictionary summarizing file counts per emotion and total counts.
    """
    summary = {}

    # Count files in each directory
    face_counts = count_files(face_dir)
    spec_counts = count_files(spec_dir)

    total_faces = 0
    total_spectrograms = 0

    # Union of all emotion categories found in either folder
    all_emotions = set(face_counts.keys()).union(set(spec_counts.keys()))

    for emotion in all_emotions:
        face_count = face_counts.get(emotion, 0)
        spec_count = spec_counts.get(emotion, 0)
        total_faces += face_count
        total_spectrograms += spec_count

        # Store per-class counts
        summary[emotion] = {
            "face_images": face_count,
            "spectrograms": spec_count
        }

    # Add global totals to the summary
    summary["TOTAL"] = {
        "face_images": total_faces,
        "spectrograms": total_spectrograms
    }

    return summary


summary = count_images_and_spectrograms()

# Print results in readable format
for emotion, counts in summary.items():
    print(f"{emotion}: {counts['face_images']} face images, {counts['spectrograms']} spectrograms")
