import os
import re
import numpy as np
import pandas as pd
from Config import config


class MapProcessor:
    def __init__(self, base_path, output_path, video_rate=config.VIDEO_RATE, debug=config.DEBUG):
        """
        Initialize the MapProcessor with base and output paths.

        Args:
            base_path (str): Base path for the dataset.
            output_path (str): Output path for saving processed data.
            video_rate (int): Video frame rate for the dataset.
            debug (bool): If True, enables debug messages.
        """
        self.base_path = base_path
        self.output_path = output_path
        self.video_rate = video_rate
        self.debug = debug
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def find_speaker(identifiers, file_name):
        """
        Determine whether the speaker is on the right or left side based on session and identifier.

        Args:
            identifiers (list): List of identifiers for each frame.
            file_name (str): Name of the file to determine the lead speaker.

        Returns:
            list: List of speaker positions ('L' for left, 'R' for right).
        """
        speaker = []
        if file_name[5] == 'F':  # If the lead speaker is a woman
            for identifier in identifiers:
                speaker.append('R' if identifier[-4] == 'M' else 'L')  # Right for male, Left for female
        elif file_name[5] == 'M':  # If the lead speaker is a man
            for identifier in identifiers:
                speaker.append('L' if identifier[-4] == 'M' else 'R')  # Left for male, Right for female
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
                # Resolve overlap by swapping frame boundaries
                last_frames[i], first_frames[i + 1] = first_frames[i + 1], last_frames[i]

        return first_frames, last_frames

    def save_data_frame(self, first_frame, last_frame, identifier, emotion, emotion_vector, speakers, file_name,
                        start_times, end_times):
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
            start_times (list): List of start times.
            end_times (list): List of end times.

        Returns:
            pd.DataFrame: The sorted and processed DataFrame.
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

        # Remove rows where emotion is 'xxx' (invalid or uninterpretable emotion)
        df = df[df['emotion'] != 'xxx']

        # Sort DataFrame by start_frame
        sorted_df = df.sort_values(by='start_frame').reset_index(drop=True)

        # Resolve overlapping frames between speakers
        updated_frames = self.cut_dual_frames(
            sorted_df['start_frame'].to_numpy(),
            sorted_df['end_frame'].to_numpy(),
            sorted_df['speaker'].to_numpy()
        )

        # Update the DataFrame with adjusted frames
        sorted_df['start_frame'] = updated_frames[0]
        sorted_df['end_frame'] = updated_frames[1]

        # Save the processed DataFrame to a CSV file
        output_file = os.path.join(self.output_path, f'{file_name}.csv')
        sorted_df.to_csv(output_file, index=False)

        if self.debug:
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
        video_rate = self.video_rate

        with open(path, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        start_times, end_times, identifiers, emotions, emotion_vectors = [], [], [], [], []

        for line in lines:
            match = re.match(pattern, line)
            if match:
                start_times.append(float(match.group(1)))  # Extract start time
                end_times.append(float(match.group(2)))  # Extract end time
                identifiers.append(match.group(3))  # Extract speaker identifier
                emotions.append(match.group(4))  # Extract emotion label
                # Extract emotion vector as a list of floats
                emotion_vectors.append(list(map(float, match.group(5).split(", "))))

        # Convert times to frame indices
        start_frames = np.add(np.array(np.multiply(start_times, video_rate), dtype=int), 1)
        end_frames = np.array(np.multiply(end_times, video_rate), dtype=int)

        # Determine speaker positions
        speakers = self.find_speaker(identifiers, file_name)

        # Save the processed data to a DataFrame and export to CSV
        self.save_data_frame(start_frames, end_frames, identifiers, emotions, emotion_vectors, speakers, file_name, start_times, end_times)

    def process_sessions(self, n_sessions):
        """
        Process all sessions in the dataset.

        Args:
            n_sessions (int): Number of sessions to process.

        Returns:
            None
        """
        for ses in range(1, n_sessions + 1):
            # Path to session evaluation files
            evaluation_path = os.path.join(self.base_path, f'Session{ses}/dialog/EmoEvaluation/')
            # Get all relevant text files
            evaluations = [file for file in os.listdir(evaluation_path) if file.endswith('.txt') and not file.startswith('._')]

            for eval_file in evaluations:
                # Process each evaluation file
                self.extract_values(os.path.join(evaluation_path, eval_file), eval_file[:-4])


# Instantiate and run the processor
processor = MapProcessor(
    base_path=config.BASE_PATH,
    output_path=config.MAP_PATH
)
processor.process_sessions(n_sessions=config.N_SESSIONS)
