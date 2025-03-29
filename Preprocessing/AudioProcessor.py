import os
import cv2
import numpy as np
import pandas as pd
import logging
import librosa
from concurrent.futures import ProcessPoolExecutor
from scipy.io import wavfile
from Config import config

# Logging configuration
logging.basicConfig(level=logging.DEBUG if config.DEBUG else logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global configuration
BASE_DIR = config.BASE_PATH
MAP_DIR = config.MAP_PATH
OUTPUT_DIR = config.SPECTROGRAM_PATH
FFT_SIZE = config.FFT_SIZE
STEP_SIZE = config.STEP_SIZE
SPEC_THRESHOLD = config.SPEC_THRESHOLD
VIDEO_RATE = config.VIDEO_RATE


def create_spectrogram(data, fft_size=FFT_SIZE, step_size=STEP_SIZE, threshold=SPEC_THRESHOLD, log=True):
    """
    Generate a spectrogram from raw audio using librosa's STFT.

    This function computes the short-time Fourier transform (STFT), converts it
    to a decibel scale (dB), applies thresholding, and normalizes the result to
    a 0–255 range suitable for image representation (as grayscale).

    Args:
        data (np.ndarray): 1D audio signal.
        fft_size (int): FFT window size.
        step_size (int): Hop length between windows.
        threshold (float): Minimum dB threshold to clip the output.
        log (bool): Whether to apply log scaling (dB).

    Returns:
        np.ndarray: Spectrogram image as a 2D uint8 array (0–255 grayscale).
    """
    """
    Create a normalized spectrogram using librosa's STFT and convert it to 0-255 grayscale image.
    """
    try:
        spec = librosa.stft(data.astype(float), n_fft=fft_size, hop_length=int(step_size))
        spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        if log:
            spec_db[spec_db < -threshold] = -threshold
        else:
            spec_db[spec_db < threshold] = threshold

        # Normalize to 0-255
        norm_spec = ((spec_db - spec_db.min()) / (spec_db.max() - spec_db.min()) * 255).astype(np.uint8)
        return norm_spec
    except Exception as e:
        logging.exception("Error creating spectrogram with librosa: %s", e)
        return np.zeros((fft_size//2 + 1, 1), dtype=np.uint8)


def extract_audio_segments(audio_path, start_frames, end_frames, video_rate=VIDEO_RATE):
    """
    Extracts audio segments corresponding to specific frame ranges.

    For each frame in the range, computes the corresponding audio time window
    based on the video frame rate and extracts that chunk of audio data.

    Args:
        audio_path (str): Path to the audio file (.wav).
        start_frames (list of int): List of starting frame indices.
        end_frames (list of int): List of ending frame indices.
        video_rate (int): Frame rate of the original video (used to map frames to audio time).

    Returns:
        dict: Dictionary mapping index -> list of 1D audio chunks (numpy arrays).
    """
    logging.info(f"Extracting audio segments from file: {audio_path}")
    audio_rate, data = wavfile.read(audio_path)

    if data is None or len(data) == 0:
        logging.error(f"Cannot read audio file: {audio_path}")
        return None

    extracted_segments = {}
    for i, (start, end) in enumerate(zip(start_frames, end_frames)):
        segments = []
        for frame in range(start, end + 1):
            segment_start = int(frame * audio_rate / video_rate)
            segment_end = int((frame + 1) * audio_rate / video_rate)

            if segment_start >= len(data) or segment_end > len(data):
                logging.warning(f"Invalid segment range ({segment_start}, {segment_end}) for frame {frame} in audio file: {audio_path}")
                continue

            segments.append(data[segment_start:segment_end])
        extracted_segments[i] = segments

    logging.info(f"Extracted {len(extracted_segments)} segments from audio file: {audio_path}")
    return extracted_segments


def generate_spectrogram_images(audio_segments, metadata):
    """
    Generate and save spectrogram images for each audio segment.

    For every time range (indexed by dict key), iterates over its segments,
    generates a spectrogram using librosa, normalizes it to 0–255, applies a
    color map (COLORMAP_JET), and saves the result as a PNG image.

    Args:
        audio_segments (dict): Mapping from index to list of audio chunks.
        metadata (pd.DataFrame): Metadata with identifier and emotion labels.
    """
    for idx, segments in audio_segments.items():
        try:
            identifier = metadata['identifier'][idx]
            emotion = metadata['emotion'][idx]
            emotion_dir = os.path.join(OUTPUT_DIR, emotion)
            os.makedirs(emotion_dir, exist_ok=True)

            for frame_idx, segment in enumerate(segments):
                spec = create_spectrogram(segment.astype('float64'))
                filename = f"{identifier}_{frame_idx}.png"
                filepath = os.path.join(emotion_dir, filename)
                colored_spec = cv2.applyColorMap(spec, cv2.COLORMAP_JET)
                colored_spec = cv2.cvtColor(colored_spec, cv2.COLOR_BGR2RGB)
                cv2.imwrite(filepath, colored_spec)
                logging.info(f"Saved spectrogram: {filepath}")

        except Exception as e:
            logging.exception(f"Error generating spectrogram for {metadata['identifier'][idx]}: {e}")


def process_audio_file(audio_file, session_num):
    """
    Process a single audio file for a given session.

    Loads metadata with frame intervals, extracts audio segments
    corresponding to those intervals, and generates spectrogram
    images saved under the corresponding emotion category.

    Args:
        audio_file (str): Name of the audio file to process.
        session_num (int): Session number indicating folder location.
    """
    logging.info(f"Processing audio file: {audio_file} in session {session_num}")
    audio_path = f"{BASE_DIR}/Session{session_num}/dialog/wav/{audio_file}"
    map_path = f"{MAP_DIR}/{audio_file[:-4]}.csv"

    if not os.path.exists(map_path):
        logging.error(f"Map {map_path} does not exist, skipping audio file {audio_file}.")
        return

    metadata = pd.read_csv(map_path)
    start_frames = metadata['start_frame'].tolist()
    end_frames = metadata['end_frame'].tolist()

    segments = extract_audio_segments(audio_path, start_frames, end_frames)
    if segments:
        generate_spectrogram_images(segments, metadata)


def process_session_audio(n_session):
    """
    Process all audio files across multiple sessions.

    For each session, this function looks for .wav files in the expected
    directory structure, and concurrently processes each file to extract audio
    segments and generate spectrogram images. Results are saved in subdirectories
    categorized by emotion.

    Args:
        n_session (int): Number of sessions to process (1-indexed).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for session_num in range(1, n_session + 1):
        audio_dir = f"{BASE_DIR}/Session{session_num}/dialog/wav"

        if not os.path.exists(audio_dir):
            logging.error(f"Path {audio_dir} does not exist, skipping session {session_num}.")
            continue

        audio_files = [file for file in os.listdir(audio_dir) if file.endswith('.wav') and not file.startswith('._')]

        if not audio_files:
            logging.warning(f"No audio files found in {audio_dir}.")
            continue

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_audio_file, audio_file, session_num) for audio_file in audio_files]
            for future in futures:
                future.result()


if __name__ == "__main__":
    process_session_audio(config.N_SESSIONS)
    logging.info("Audio processing completed.")
