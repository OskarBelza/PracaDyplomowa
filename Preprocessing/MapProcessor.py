import os
import re
import numpy as np
import pandas as pd
from Config import config

# Global parameters from config
VIDEO_RATE = config.VIDEO_RATE
DEBUG = config.DEBUG


def find_speaker(identifiers, file_name):
    """
    Determine speaker position ('L' or 'R') based on speaker gender and session lead.

    Parameters:
        identifiers (List[str]): Speaker identifiers from metadata.
        file_name (str): Name of the transcript file.

    Returns:
        List[str]: List of 'L' or 'R' depending on speaker position per segment.
    """
    speaker = []
    if file_name[5] == 'F':  # Lead is female
        for identifier in identifiers:
            speaker.append('R' if identifier[-4] == 'M' else 'L')
    elif file_name[5] == 'M':  # Lead is male
        for identifier in identifiers:
            speaker.append('L' if identifier[-4] == 'M' else 'R')
    return speaker


def cut_dual_frames(first_frames, last_frames, speakers):
    """
    Fix overlapping segments between two speakers in a conversation.

    Parameters:
        first_frames (np.ndarray): Array of starting frame indices.
        last_frames (np.ndarray): Array of ending frame indices.
        speakers (np.ndarray): Array of speaker labels ('L' or 'R').

    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted start and end frame arrays.
    """
    first_frames = first_frames.copy()
    last_frames = last_frames.copy()

    for i in range(len(first_frames) - 1):
        if speakers[i] != speakers[i + 1] and last_frames[i] > first_frames[i + 1]:
            last_frames[i], first_frames[i + 1] = first_frames[i + 1], last_frames[i]

    return first_frames, last_frames


def save_data_frame(first_frame, last_frame, identifier, emotion, emotion_vector, speakers,
                    file_name, start_times, end_times, output_path):
    """
    Assemble a DataFrame from extracted values, clean and save it to CSV.

    Parameters:
        first_frame (List[int]): Starting frame indices.
        last_frame (List[int]): Ending frame indices.
        identifier (List[str]): Speaker IDs.
        emotion (List[str]): Emotion labels.
        emotion_vector (List[List[float]]): Emotion vector values.
        speakers (List[str]): Speaker positions ('L' or 'R').
        file_name (str): Output filename base (without extension).
        start_times (List[float]): Start times in seconds.
        end_times (List[float]): End times in seconds.
        output_path (str): Directory to save CSV files.

    Returns:
        pd.DataFrame: Cleaned and saved DataFrame.
    """
    data = {
        'start_frame': first_frame,
        'end_frame': last_frame,
        'identifier': identifier,
        'emotion': emotion,
        'emotion_vector': emotion_vector,
        'speaker': speakers,
        'start_time': start_times,
        'end_time': end_times
    }
    df = pd.DataFrame(data)
    df = df[df['emotion'] != 'xxx']
    sorted_df = df.sort_values(by='start_frame').reset_index(drop=True)

    adjusted_start, adjusted_end = cut_dual_frames(
        sorted_df['start_frame'].to_numpy(),
        sorted_df['end_frame'].to_numpy(),
        sorted_df['speaker'].to_numpy()
    )

    sorted_df['start_frame'] = adjusted_start
    sorted_df['end_frame'] = adjusted_end

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'{file_name}.csv')
    sorted_df.to_csv(output_file, index=False)

    if DEBUG:
        print(f"Saved map CSV to {output_file}")

    return sorted_df


def extract_values(path, file_name, output_path, video_rate=VIDEO_RATE):
    """
    Extract annotation information from raw evaluation file and save as structured CSV.

    Parameters:
        path (str): Path to the transcript text file.
        file_name (str): File name without extension, used for output.
        output_path (str): Path to output folder where CSV will be saved.
        video_rate (int): Frame rate of the original video data.

    Returns:
        None
    """
    pattern = r"\[(.*?) - (.*?)\]\t(\S+)\t(\S+)\t\[(.*?)\]"

    with open(path, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    start_times, end_times, identifiers, emotions, emotion_vectors = [], [], [], [], []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            start_times.append(float(match.group(1)))
            end_times.append(float(match.group(2)))
            identifiers.append(match.group(3))
            emotions.append(match.group(4))
            emotion_vectors.append(list(map(float, match.group(5).split(", "))))

    start_frames = np.add(np.array(np.multiply(start_times, video_rate), dtype=int), 1)
    end_frames = np.array(np.multiply(end_times, video_rate), dtype=int)

    speakers = find_speaker(identifiers, file_name)

    save_data_frame(
        start_frames, end_frames, identifiers, emotions,
        emotion_vectors, speakers, file_name,
        start_times, end_times, output_path
    )


def process_sessions(base_path, output_path, n_sessions):
    """
    Process all annotation files from multiple sessions and save structured CSVs.

    Parameters:
        base_path (str): Base dataset directory (e.g. IEMOCAP root path).
        output_path (str): Directory where CSV outputs will be saved.
        n_sessions (int): Number of sessions to process.

    Returns:
        None
    """
    for ses in range(1, n_sessions + 1):
        evaluation_path = os.path.join(base_path, f'Session{ses}/dialog/EmoEvaluation/')
        if not os.path.exists(evaluation_path):
            print(f"[Warning] Path not found: {evaluation_path}")
            continue

        files = [f for f in os.listdir(evaluation_path) if f.endswith('.txt') and not f.startswith('._')]

        for file in files:
            full_path = os.path.join(evaluation_path, file)
            extract_values(full_path, file[:-4], output_path)


if __name__ == "__main__":
    process_sessions(
        base_path=config.BASE_PATH,
        output_path=config.MAP_PATH,
        n_sessions=config.N_SESSIONS
    )
