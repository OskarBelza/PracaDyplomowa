import os
import re
import numpy as np
import pandas as pd
import config


class IEMOCAPDataMap:
    def __init__(self, base_path, output_path):
        """
        Initialize the IEMOCAPProcessor with base and output paths.

        Args:
            base_path (str): Base path for the IEMOCAP dataset.
            output_path (str): Output path for saving processed data.
        """
        self.base_path = base_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def find_speaker(identifiers, file_name):
        """
        Determine whether the speaker is on the right or left side based on session and identifier.

        Args:
            identifiers (list): List of identifiers for each frame.
            file_name (str): Name of the file to determine lead speaker.

        Returns:
            list: List of speaker positions ('L' for left, 'R' for right).
        """
        speaker = []
        if file_name[5] == 'F':  # Woman is lead speaker
            for i, identifier in enumerate(identifiers):
                speaker.append('R' if identifier[-4] == 'M' else 'L')
        elif file_name[5] == 'M':  # Man is lead speaker
            for i, identifier in enumerate(identifiers):
                speaker.append('L' if identifier[-4] == 'M' else 'R')
        return speaker

    @staticmethod
    def cut_dual_frames(first_frames, last_frames, speakers):
        """
        Adjust frames to remove overlapping segments between different speakers.

        Args:
            first_frames (numpy.ndarray): Array of first frames.
            last_frames (numpy.ndarray): Array of last frames.
            speakers (numpy.ndarray): Array of speaker identifiers.

        Returns:
            tuple: Adjusted arrays of first_frames and last_frames.
        """
        first_frames = first_frames.copy()
        last_frames = last_frames.copy()

        for i in range(len(first_frames) - 1):
            if speakers[i] != speakers[i + 1] and last_frames[i] > first_frames[i + 1]:
                last_frames[i], first_frames[i + 1] = first_frames[i + 1], last_frames[i]

        return first_frames, last_frames

    def save_data_frame(self, first_frame, last_frame, identifier, emotion, emotion_vector, speakers, file_name):
        """
        Save data to a sorted DataFrame and handle overlapping frames, then export to CSV.

        Args:
            first_frame (list): List of first frames.
            last_frame (list): List of last frames.
            identifier (list): List of identifiers.
            emotion (list): List of emotions.
            emotion_vector (list): List of emotion vectors.
            speakers (list): List of speakers.
            file_name (str): Name of the output file (without extension).

        Returns:
            pd.DataFrame: The sorted and processed DataFrame.
        """
        data = {
            'first_frame': first_frame,
            'last_frame': last_frame,
            'identifier': identifier,
            'emotion': emotion,
            'emotion_vector': emotion_vector,
            'speaker': speakers
        }
        df = pd.DataFrame(data)
        sorted_df = df.sort_values(by='first_frame').reset_index(drop=True)

        updated_frames = self.cut_dual_frames(
            sorted_df['first_frame'].to_numpy(),
            sorted_df['last_frame'].to_numpy(),
            sorted_df['speaker'].to_numpy()
        )

        sorted_df['first_frame'] = updated_frames[0]
        sorted_df['last_frame'] = updated_frames[1]

        output_file = os.path.join(self.output_path, f'{file_name}.csv')
        sorted_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

        return sorted_df

    def extract_values(self, path, file_name):
        """
        Extract features from the file and process them into frames and speaker data.

        Args:
            path (str): Path to the file to extract values from.
            file_name (str): Name of the file to process.

        Returns:
            None
        """
        pattern = r"\[(.*?) - (.*?)\]\t(\S+)\t(\S+)\t\[(.*?)\]"
        video_rate = 30

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

        first_frames = np.add(np.array(np.multiply(start_times, video_rate), dtype=int), 1)
        last_frames = np.array(np.multiply(end_times, video_rate), dtype=int)

        speakers = self.find_speaker(identifiers, file_name)

        self.save_data_frame(first_frames, last_frames, identifiers, emotions, emotion_vectors, speakers, file_name)

    def process_sessions(self, n_sessions):
        """
        Process all sessions in the dataset.

        Args:
            n_sessions (int): Number of sessions to process.

        Returns:
            None
        """
        for ses in range(1, n_sessions + 1):
            evaluation_path = os.path.join(self.base_path, f'Session{ses}/dialog/EmoEvaluation/')
            evaluations = [file for file in os.listdir(evaluation_path) if file.endswith('.txt') and not file.startswith('._')]

            for eval_file in evaluations:
                self.extract_values(os.path.join(evaluation_path, eval_file), eval_file[:-4])


# Example Usage
base_path = config.BASE_PATH
output_path = config.MAPA_PATH
processor = IEMOCAPDataMap(base_path, output_path)
processor.process_sessions(n_sessions=config.N_SESSIONS)
