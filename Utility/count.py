import os
from Config.config import FACE_PATH, SPECTROGRAM_PATH


def count_files(directory):
    emotion_counts = {}
    if os.path.exists(directory):
        for emotion in os.listdir(directory):
            emotion_path = os.path.join(directory, emotion)
            if os.path.isdir(emotion_path):
                num_files = len([f for f in os.listdir(emotion_path) if f.endswith('.png') or f.endswith('.npy')])
                emotion_counts[emotion] = num_files
    return emotion_counts


def count_images_and_spectrograms(face_dir=FACE_PATH, spec_dir=SPECTROGRAM_PATH):
    """
    Counts the number of face images and spectrograms for each emotion category and provides a total summary.

    Args:
        face_dir (str): Main directory containing detected face images.
        spec_dir (str): Main directory containing generated spectrograms.

    Returns:
        dict: A dictionary containing the number of files for each emotion and overall totals.
    """
    summary = {}

    # Count face images and spectrograms
    face_counts = count_files(face_dir)
    spec_counts = count_files(spec_dir)

    # Combine results into a single summary
    total_faces = 0
    total_spectrograms = 0
    all_emotions = set(face_counts.keys()).union(set(spec_counts.keys()))

    for emotion in all_emotions:
        face_count = face_counts.get(emotion, 0)
        spec_count = spec_counts.get(emotion, 0)
        total_faces += face_count
        total_spectrograms += spec_count

        summary[emotion] = {
            "face_images": face_count,
            "spectrograms": spec_count
        }

    # Add total counts
    summary["TOTAL"] = {
        "face_images": total_faces,
        "spectrograms": total_spectrograms
    }

    return summary


if __name__ == '__main__':
    summary = count_images_and_spectrograms()
    for emotion, counts in summary.items():
        print(f"{emotion}: {counts['face_images']} face images, {counts['spectrograms']} spectrograms")