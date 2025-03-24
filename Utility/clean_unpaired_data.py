import os
import glob
from Config.config import SPECTROGRAM_PATH, FACE_PATH


def clean_unpaired_data(faces_dir, spectrograms_dir):
    emotions = os.listdir(faces_dir)

    for emotion in emotions:
        print(f"🧹 Cleaning class: {emotion}")
        faces_path = os.path.join(faces_dir, emotion)
        specs_path = os.path.join(spectrograms_dir, emotion)

        face_files = set(os.path.basename(f) for f in glob.glob(os.path.join(faces_path, "*.png")))
        spec_files = set(os.path.basename(f) for f in glob.glob(os.path.join(specs_path, "*.png")))

        common_files = face_files & spec_files

        faces_to_remove = face_files - common_files
        specs_to_remove = spec_files - common_files

        for f in faces_to_remove:
            os.remove(os.path.join(faces_path, f))
            print(f"Removed face: {f}")

        for s in specs_to_remove:
            os.remove(os.path.join(specs_path, s))
            print(f"Removed spectrogram: {s}")

    print("Dataset cleaned – only paired files remain.")


clean_unpaired_data(FACE_PATH, SPECTROGRAM_PATH)
