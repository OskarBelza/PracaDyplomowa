import pandas as pd
from pathlib import Path
from Config import config

# Mapping emotions to numeric labels
emotions_dict = {
    'ang': 0,  # Anger
    'hap': 1,  # Happiness
    'exc': 2,  # Excitement
    'sad': 3,  # Sadness
    'fru': 4,  # Frustration
    'fea': 5,  # Fear
    'sur': 6,  # Surprise
    'neu': 7,  # Neutral
}

# Define paths for data
faces_dir = Path(config.FACE_PATH)
spectrograms_dir = Path(config.SPECTROGRAM_PATH)
metadata_dir = Path(config.MAP_PATH)

# Initialize list to store processed data and a counter for missing faces
all_data = []
missing_faces_count = 0

# Iterate over each session directory
for session_dir in faces_dir.iterdir():
    if not session_dir.is_dir():
        continue  # Skip if not a directory

    session_name = session_dir.name

    # Iterate over each clip directory within a session
    for clip_dir in session_dir.iterdir():
        if not clip_dir.is_dir():
            continue  # Skip if not a directory

        clip_name = clip_dir.name
        clip_data = []  # Temporary list to store data for this clip

        # Load face images and extract metadata
        for face_file in clip_dir.glob("*.jpg"):
            frame_number = int(face_file.stem.split('_')[-1])  # Extract frame number
            emotion = face_file.stem.split('_')[-2]  # Extract emotion label
            if emotion not in emotions_dict:
                continue  # Skip if emotion is not in the dictionary
            clip_data.append({
                'frame_number': frame_number,
                'face_path': str(face_file),
                'spectrogram_path': None,  # Placeholder for spectrogram path
                'emotion_label': emotions_dict[emotion],  # Map emotion to numeric label
            })

        # Load spectrogram files and match them with face frames
        spectrogram_files = list((spectrograms_dir / session_name / clip_name).glob("*.npy"))
        for spectrogram_file in spectrogram_files:
            frame_number = int(spectrogram_file.stem.split('_')[-1])  # Extract frame number
            match = next((item for item in clip_data if item['frame_number'] == frame_number), None)
            if match:
                match['spectrogram_path'] = str(spectrogram_file)  # Assign spectrogram path
            else:
                # Increment counter for missing faces
                missing_faces_count += 1

        # Add all clip data to the main dataset
        all_data.extend(clip_data)

# Create a DataFrame for easier analysis and storage
data_df = pd.DataFrame(all_data)

# Save the prepared data to a CSV file
data_df.to_csv(config.MULTIMODAL_DATA_PATH, index=False)

# Print summary of the processing
print(f"Data has been prepared and saved. Number of missing faces: {missing_faces_count}")
print(f"Sample data: {data_df.head()}")
print(f"Total number of entries: {len(data_df)}")
