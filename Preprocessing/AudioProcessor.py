import os
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
    def __init__(self, spec_h5_path, base_dir, map_dir, fft_size=config.FFT_SIZE, step_size=config.STEP_SIZE,
                 spec_threshold=config.SPEC_THRESHOLD, video_rate=config.VIDEO_RATE, debug=config.DEBUG):
        """
        Initialize the AudioProcessor with parameters for FFT and spectrogram generation.

        Args:
            spec_h5_path (str): Path to save spectrograms in HDF5 format.
            base_dir (str): Base directory for the IEMOCAP dataset.
            map_dir (str): Directory containing extraction maps.
            fft_size (int): Window size for FFT.
            step_size (float): Step size for sliding FFT window.
            spec_threshold (int): Threshold for spectrogram filtering.
            video_rate (int): Frame rate of the video.
            debug (bool): If True, enables debug messages.
        """
        self.spec_h5_path = spec_h5_path
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

    def generate_spectrogram(self, audio_segments, metadata, h5_file):
        """
        Generate spectrograms for audio segments and save them to an HDF5 file.

        Args:
            audio_segments (dict): Dictionary where keys are indices and values are lists of audio frames.
            metadata (pd.DataFrame): DataFrame containing speaker, identifier, and emotion metadata.
            h5_file (h5py.File): HDF5 file to save spectrograms.
        """
        for dict_idx, segments in audio_segments.items():
            identifier = metadata['identifier'][dict_idx]
            emotion = metadata['emotion'][dict_idx]
            group_name = f"{identifier}_{emotion}"

            if group_name not in h5_file:
                h5_file.create_group(group_name)

            for idx, frame_data in enumerate(segments):
                try:
                    spectrogram = self.pretty_spectrogram(frame_data.astype('float64'))
                    h5_file[group_name].create_dataset(str(idx), data=spectrogram, compression="gzip")
                except Exception as e:
                    logging.error(f"Error generating spectrogram for frame {idx} in identifier {identifier}: {e}")

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

    def process_audio_file(self, audio_file, session_num, h5_file):
        """
        Process a single audio file for a given session.

        Args:
            audio_file (str): Name of the audio file to process.
            session_num (int): Session number.
            h5_file (h5py.File): HDF5 file to save spectrograms.
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
            self.generate_spectrogram(extracted_frames, metadata, h5_file)

    def process_session_audio(self, n_session):
        """
        Process all audio files in a session and save spectrograms to an HDF5 file.

        Args:
            n_session (int): Number of the session to process.
        """
        with h5py.File(self.spec_h5_path, 'w') as h5_file:
            for session_num in range(1, n_session + 1):
                audio_path = f"{self.base_dir}/Session{session_num}/dialog/wav"

                if not os.path.exists(audio_path):
                    logging.error(f"Path {audio_path} does not exist, skipping session {session_num}.")
                    continue

                audio_files = [file for file in os.listdir(audio_path) if file.endswith('.wav') and not file.startswith('._')]

                logging.info(f"Starting audio processing for session {session_num} with {len(audio_files)} files.")

                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.process_audio_file, audio_file, session_num, h5_file) for audio_file in audio_files]
                    for future in futures:
                        future.result()

        logging.info(f"Completed processing for session {n_session}")


# Instantiate and start processing sessions
audio_processor = AudioProcessor(
    spec_h5_path=config.SPECTROGRAM_PATHH5,
    base_dir=config.BASE_PATH,
    map_dir=config.MAP_PATH,
    debug=config.DEBUG
)

audio_processor.process_session_audio(n_session=config.N_SESSIONS)

logging.info("Processing completed.")
