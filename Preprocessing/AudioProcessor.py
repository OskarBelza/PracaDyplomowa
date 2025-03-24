import os

import cv2
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from Config import config
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging
import h5py

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG if config.DEBUG else logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class AudioProcessor:
    def __init__(self, output_dir, base_dir, map_dir, fft_size=config.FFT_SIZE, step_size=config.STEP_SIZE,
                 spec_threshold=config.SPEC_THRESHOLD, video_rate=config.VIDEO_RATE, debug=config.DEBUG):
        """
        Initialize the AudioProcessor with parameters for FFT and spectrogram generation.

        Args:
            output_dir (str): Path to the output directory.
            base_dir (str): Base directory for the IEMOCAP dataset.
            map_dir (str): Directory containing extraction maps.
            fft_size (int): Window size for FFT.
            step_size (float): Step size for sliding FFT window.
            spec_threshold (int): Threshold for spectrogram filtering.
            video_rate (int): Frame rate of the video.
            debug (bool): If True, enables debug messages.
        """
        self.output_dir = output_dir
        self.base_dir = base_dir
        self.map_dir = map_dir
        self.fft_size = fft_size
        self.step_size = step_size
        self.spec_threshold = spec_threshold
        self.video_rate = video_rate
        self.debug = debug

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        """
        Create a Butterworth bandpass filter.

        Args:
            lowcut (float): Low cut-off frequency in Hz.
            highcut (float): High cut-off frequency in Hz.
            fs (int): Sampling frequency in Hz.
            order (int): Order of the filter.

        Returns:
            tuple: Numerator (b) and denominator (a) polynomials of the filter.
        """
        nyq = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a Butterworth bandpass filter to the data.

        Args:
            data (np.ndarray): Input signal.
            lowcut (float): Low cut-off frequency in Hz.
            highcut (float): High cut-off frequency in Hz.
            fs (int): Sampling frequency in Hz.
            order (int): Order of the filter.

        Returns:
            np.ndarray: Filtered signal.
        """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def stft(X, fftsize=128, step=65, mean_normalize=True):
        """
        Compute Short-Time Fourier Transform (STFT).

        Args:
            X (np.ndarray): Input signal.
            fftsize (int): Size of the FFT window.
            step (int): Step size for sliding the window.
            mean_normalize (bool): Whether to normalize the signal.

        Returns:
            np.ndarray: Complex STFT result.
        """
        if mean_normalize:
            X -= X.mean()  # Remove mean for normalization

        # Pad the signal to ensure compatibility with FFT size
        X = np.pad(X, (0, fftsize - len(X) % fftsize), mode='constant')
        noverlap = fftsize - step
        shape = (len(X) - fftsize + 1, fftsize)
        strides = (X.strides[0], X.strides[0])
        windows = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)[::step]
        win = np.hanning(fftsize)  # Apply a Hann window
        return np.fft.rfft(windows * win, n=fftsize)

    def pretty_spectrogram(self, d, log=True):
        """
        Create a spectrogram from audio data.

        Args:
            d (np.ndarray): Input signal.
            log (bool): Whether to apply a logarithmic scale.

        Returns:
            np.ndarray: Spectrogram.
        """
        specgram = np.abs(self.stft(d, fftsize=self.fft_size, step=int(self.step_size)))

        if log:
            specgram /= specgram.max()
            specgram = np.log10(specgram + 1e-10)  # Avoid log of zero
            specgram[specgram < -self.spec_threshold] = -self.spec_threshold
        else:
            specgram[specgram < self.spec_threshold] = self.spec_threshold

        return specgram

    def generate_spectrogram(self, audio_segments, metadata):
        """
        Generate spectrograms for audio segments and save them to an HDF5 file.

        Args:
            audio_segments (dict): Dictionary where keys are indices and values are lists of audio frames.
            metadata (pd.DataFrame): DataFrame containing speaker, identifier, and emotion metadata.
        """
        for dict_idx, segments in audio_segments.items():
            try:
                identifier = metadata['identifier'][dict_idx]
                emotion = metadata['emotion'][dict_idx]

                emotion_dir = os.path.join(self.output_dir, emotion)
                os.makedirs(emotion_dir, exist_ok=True)

                for idx, frame_data in enumerate(segments):
                    spectrogram = self.pretty_spectrogram(frame_data.astype('float64'))

                    # Normalizacja wartości do zakresu 0-255 (konwersja do obrazu)
                    spectrogram = ((spectrogram - spectrogram.min()) / (
                                spectrogram.max() - spectrogram.min()) * 255).astype(np.uint8)

                    # Zapis spektrogramu jako obraz PNG
                    filename = f"{identifier}_{idx}.png"
                    filepath = os.path.join(emotion_dir, filename)
                    cv2.imwrite(filepath, spectrogram)

                    logging.info(f"Zapisano spektrogram: {filepath}")

            except Exception as e:
                logging.exception(f"Błąd podczas generowania spektrogramu dla {identifier}: {e}")

    def extract_audio_segments_from_ranges(self, audio_path, start_frames, end_frames):
        """
        Extract audio frames from ranges and group them under the corresponding range index.

        Args:
            audio_path (str): Path to the audio file.
            start_frames (list): List of start frame indices.
            end_frames (list): List of end frame indices.

        Returns:
            dict: Dictionary where the key is the range index and the value is a list of audio frames.
        """
        logging.info(f"Extracting audio segments from file: {audio_path}")
        audio_rate, data = wavfile.read(audio_path)

        if data is None or len(data) == 0:
            logging.error(f"Cannot read audio file: {audio_path}")
            return None

        extracted_segments = {}
        for i, (start, end) in enumerate(zip(start_frames, end_frames)):
            frames = []
            logging.debug(f"Processing audio range {start}-{end} for index {i}")
            for frame in range(start, end + 1):
                segment_start = int(frame * audio_rate / self.video_rate)
                segment_end = int((frame + 1) * audio_rate / self.video_rate)

                if segment_start >= len(data) or segment_end > len(data):
                    logging.warning(f"Invalid segment range ({segment_start}, {segment_end}) for frame {frame} in audio file: {audio_path}")
                    continue

                frames.append(data[segment_start:segment_end])
            extracted_segments[i] = frames

        logging.info(f"Extracted {len(extracted_segments)} segments from audio file: {audio_path}")
        return extracted_segments

    def process_audio_file(self, audio_file, session_num):
        """
        Process a single audio file for a given session.

        Args:
            audio_file (str): Name of the audio file to process.
            session_num (int): Session number.
        """
        logging.info(f"Processing audio file: {audio_file} in session {session_num}")
        audio_path = f"{self.base_dir}/Session{session_num}/dialog/wav/{audio_file}"

        map_path = f"{self.map_dir}/{audio_file[:-4]}.csv"
        if not os.path.exists(map_path):
            logging.error(f"Map {map_path} does not exist, skipping audio file {audio_file}.")
            return

        metadata = pd.read_csv(map_path)
        start_frames = metadata['start_frame'].tolist()
        end_frames = metadata['end_frame'].tolist()

        extracted_frames = self.extract_audio_segments_from_ranges(audio_path, start_frames, end_frames)
        if extracted_frames:
            self.generate_spectrogram(extracted_frames, metadata)

    def process_session_audio(self, n_session):
        """
        Process all audio files in a session and save spectrograms to an HDF5 file.

        Args:
            n_session (int): Number of the session to process.
        """
        os.makedirs(self.output_dir, exist_ok=True)  # Tworzymy katalog główny

        for session_num in range(1, n_session + 1):
            audio_path = f"{self.base_dir}/Session{session_num}/dialog/wav"

            if not os.path.exists(audio_path):
                logging.error(f"Ścieżka {audio_path} nie istnieje, pomijam sesję {session_num}.")
                continue

            audio_files = [file for file in os.listdir(audio_path) if
                           file.endswith('.wav') and not file.startswith('._')]

            if not audio_files:
                logging.warning(f"Brak plików audio w {audio_path}.")
                continue

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.process_audio_file, audio_file, session_num) for audio_file
                           in audio_files]
                for future in futures:
                    future.result()


# Instantiate and start processing sessions
audio_processor = AudioProcessor(
    output_dir=config.SPECTROGRAM_PATH,
    base_dir=config.BASE_PATH,
    map_dir=config.MAP_PATH,
    debug=config.DEBUG
)

audio_processor.process_session_audio(n_session=config.N_SESSIONS)

logging.info("Processing completed.")
