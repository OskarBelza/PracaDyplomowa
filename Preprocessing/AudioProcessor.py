import os
import cv2
import numpy as np
import pandas as pd
import logging
import librosa
from concurrent.futures import ProcessPoolExecutor
from Config import config

# Configure logging level and format depending on the DEBUG flag in config
logging.basicConfig(level=logging.DEBUG if config.DEBUG else logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global constants from configuration
BASE_DIR = config.BASE_PATH
MAP_DIR = config.MAP_PATH
OUTPUT_DIR = config.SPECTROGRAM_PATH
VIDEO_RATE = config.VIDEO_RATE
FFT_SIZE = config.FFT_SIZE
HOP_LENGTH = config.HOP_LENGTH
MELS = config.MELS
FMIN = config.FMIN


def create_mel_spectrogram(data, sr, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=MELS,
                           fmin=FMIN, fmax=None, output_size=(224, 224)):
    """
    Generates a mel spectrogram from an audio signal and resizes it to a specified shape.

    Parameters:
        data (np.ndarray): 1D array representing the raw audio waveform.
        sr (int): Sample rate of the audio.
        n_fft (int): Number of FFT components.
        hop_length (int): Hop length between frames.
        n_mels (int): Number of Mel bands to generate.
        fmin (int): Minimum frequency for Mel filterbank.
        fmax (int or None): Maximum frequency for Mel filterbank.
        output_size (tuple): Target (height, width) of the output spectrogram image.

    Returns:
        np.ndarray: 2D uint8 image representing the normalized mel spectrogram.
    """
    try:
        if len(data) == 0:
            logging.warning("Received empty audio signal.")
            return np.zeros(output_size, dtype=np.uint8)

        if len(data) < n_fft:
            data = np.pad(data, (0, n_fft - len(data)), mode='constant')

        mel_spec = librosa.feature.melspectrogram(
            y=data, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        min_val, max_val = mel_spec_db.min(), mel_spec_db.max()
        if max_val - min_val == 0:
            logging.warning("Mel spectrogram has no dynamic range. Returning blank image.")
            return np.zeros(output_size, dtype=np.uint8)

        norm_spec = ((mel_spec_db - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        resized_spec = cv2.resize(norm_spec, output_size, interpolation=cv2.INTER_AREA)

        return resized_spec

    except Exception as e:
        logging.exception("Error generating mel spectrogram: %s", e)
        return np.zeros(output_size, dtype=np.uint8)


def extract_audio_segments(data, sr, start_frames, end_frames, video_rate=VIDEO_RATE):
    """
    Splits the raw audio waveform into segments based on video frame alignment.

    Parameters:
        data (np.ndarray): The raw audio waveform.
        sr (int): Sample rate of the audio signal.
        start_frames (List[int]): List of starting video frame indices for each utterance.
        end_frames (List[int]): List of ending video frame indices for each utterance.
        video_rate (int): Video frame rate (frames per second).

    Returns:
        dict[int, List[np.ndarray]]: Dictionary mapping utterance indices to a list of segments.
    """
    if data is None or len(data) == 0:
        logging.error("No audio data available for processing.")
        return None

    extracted_segments = {}

    for i, (start, end) in enumerate(zip(start_frames, end_frames)):
        segments = []
        for frame in range(start, end + 1):
            segment_start = int(frame * sr / video_rate)
            segment_end = int((frame + 1) * sr / video_rate)

            if segment_end > len(data):
                logging.warning(f"Segment exceeds audio length at frame {frame} (utterance {i}). Skipping.")
                continue

            segment = data[segment_start:segment_end]
            segments.append(segment)

        if segments:
            extracted_segments[i] = segments

    logging.info(f"Extracted {len(extracted_segments)} utterance segments.")
    return extracted_segments


def generate_spectrogram_images(audio_segments, metadata, sr):
    """
    Creates and saves mel spectrogram images for each audio segment,
    organizing them by emotion class into folders.

    Parameters:
        audio_segments (dict): Dictionary of utterance indices mapped to segment lists.
        metadata (pd.DataFrame): DataFrame with 'identifier' and 'emotion' columns.
        sr (int): Sample rate of the audio.
    """
    for idx, segments in audio_segments.items():
        try:
            identifier = metadata['identifier'][idx]
            emotion = metadata['emotion'][idx]
            emotion_dir = os.path.join(OUTPUT_DIR, emotion)
            os.makedirs(emotion_dir, exist_ok=True)

            for frame_idx, segment in enumerate(segments):
                spec = create_mel_spectrogram(segment, sr)
                filename = f"{identifier}_{frame_idx}.png"
                filepath = os.path.join(emotion_dir, filename)

                # Apply JET colormap and convert to RGB before saving
                colored_spec = cv2.applyColorMap(spec, cv2.COLORMAP_JET)
                colored_spec = cv2.cvtColor(colored_spec, cv2.COLOR_BGR2RGB)
                cv2.imwrite(filepath, colored_spec)

                logging.info(f"Saved spectrogram: {filepath}")

        except Exception as e:
            logging.exception(f"Error generating spectrograms for utterance {idx}: {e}")


def process_audio_file(audio_file, session_num):
    """
    Full processing pipeline for one audio file:
    - Loads audio and corresponding metadata
    - Extracts segments
    - Saves spectrograms

    Parameters:
        audio_file (str): Name of the audio file (e.g. "Ses01F_script01_1.wav").
        session_num (int): Session number (e.g. 1, 2, ..., 5).
    """
    logging.info(f"Processing audio file: {audio_file} (Session {session_num})")

    audio_path = f"{BASE_DIR}/Session{session_num}/dialog/wav/{audio_file}"
    map_path = f"{MAP_DIR}/{audio_file[:-4]}.csv"

    if not os.path.exists(map_path):
        logging.error(f"Metadata CSV {map_path} not found. Skipping file.")
        return

    metadata = pd.read_csv(map_path)
    start_frames = metadata['start_frame'].tolist()
    end_frames = metadata['end_frame'].tolist()

    data, sr = librosa.load(audio_path, sr=None)
    segments = extract_audio_segments(data, sr, start_frames, end_frames)

    if segments:
        generate_spectrogram_images(segments, metadata, sr)


def process_session_audio(n_session):
    """
    Iterates through all sessions and processes each audio file in parallel using a process pool.

    Parameters:
        n_session (int): Number of sessions to process (e.g. 5 for IEMOCAP).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for session_num in range(1, n_session + 1):
        audio_dir = f"{BASE_DIR}/Session{session_num}/dialog/wav"

        if not os.path.exists(audio_dir):
            logging.error(f"Directory {audio_dir} not found. Skipping session {session_num}.")
            continue

        audio_files = [
            file for file in os.listdir(audio_dir)
            if file.endswith('.wav') and not file.startswith('._')
        ]

        if not audio_files:
            logging.warning(f"No .wav files found in {audio_dir}.")
            continue

        # Run file processing in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_audio_file, audio_file, session_num) for audio_file in audio_files]
            for future in futures:
                future.result()


if __name__ == "__main__":
    # Entry point: process all sessions defined in config
    process_session_audio(config.N_SESSIONS)
    logging.info("Audio processing completed.")
