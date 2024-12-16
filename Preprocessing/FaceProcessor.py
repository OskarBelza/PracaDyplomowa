import os
import cv2
import dlib
import numpy as np
import pandas as pd
import h5py
from Config import config
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG if config.DEBUG else logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FaceProcessor:
    def __init__(self, face_h5_path, base_dir, map_dir, y1=config.Y1, y2=config.Y2, x1=config.X1, x2=config.X2,
                 y3=config.Y3, y4=config.Y4, x3=config.X3, x4=config.X4, face_size=config.IMAGE_SIZE, debug=config.DEBUG):
        """
        Initialize the FaceProcessor with directory paths and cropping coordinates.

        Args:
            face_h5_path (str): Path to the HDF5 file to save cropped face images.
            base_dir (str): Directory containing video files.
            map_dir (str): Directory containing extraction maps.
            y1, y2, x1, x2, y3, y4, x3, x4: Coordinates for cropping frames based on speaker.
            face_size (int): Size of the cropped face images.
            debug (bool): If True, enables debug messages.
        """
        self.face_h5_path = face_h5_path
        self.base_dir = base_dir
        self.map_dir = map_dir
        self.y1, self.y2, self.x1, self.x2 = y1, y2, x1, x2
        self.y3, self.y4, self.x3, self.x4 = y3, y4, x3, x4
        self.face_size = face_size
        self.debug = debug
        self.detector = dlib.get_frontal_face_detector()

    def cut_frame(self, frame, speaker):
        """
        Crop the frame based on the speaker's position.

        Args:
            frame (numpy.ndarray): The original frame.
            speaker (str): The speaker's position ('L' for left, 'R' for right).

        Returns:
            numpy.ndarray: Cropped frame for the specified speaker.
        """
        if speaker == 'L':
            return frame[self.y1:self.y2, self.x1:self.x2]
        else:
            return frame[self.y3:self.y4, self.x3:self.x4]

    @staticmethod
    def extract_frames_from_ranges(video_path, start_frames, end_frames):
        """
        Extract frames from a video based on given ranges.

        Args:
            video_path (str): Path to the video file.
            start_frames (list): List of start frame indices.
            end_frames (list): List of end frame indices.

        Returns:
            dict: Dictionary where keys are range indices and values are lists of frames.
        """
        logging.info(f"Extracting frames from video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logging.error(f"Cannot open video: {video_path}")
            return None

        extracted_frames = {}
        for i, (start, end) in enumerate(zip(start_frames, end_frames)):
            frames_in_range = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            logging.debug(f"Processing frame range {start}-{end} from video: {video_path}")

            for frame_idx in range(start, end + 1):
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Failed to read frame {frame_idx} from video: {video_path}")
                    break
                frames_in_range.append(frame)

            extracted_frames[i] = frames_in_range

        cap.release()
        logging.info(f"Extracted frames for {len(extracted_frames)} ranges from video: {video_path}")
        return extracted_frames

    def detect_face(self, frames_dict, metadata, h5_file):
        """
        Detect faces in video frames, crop them, and save to an HDF5 file.

        Args:
            frames_dict (dict): Dictionary where keys are indices and values are lists of frames.
            metadata (pd.DataFrame): DataFrame containing speaker, identifier, and emotion metadata.
            h5_file (h5py.File): HDF5 file handle to save cropped face images.
        """
        logging.info("Starting face detection and saving to HDF5.")

        for dict_idx, frames in frames_dict.items():
            speaker = metadata['speaker'][dict_idx]
            identifier = metadata['identifier'][dict_idx]
            emotion = metadata['emotion'][dict_idx]
            first_frame = metadata['start_frame'][dict_idx]
            last_frame = metadata['end_frame'][dict_idx]

            group_name = f"{identifier}_{emotion}"
            if group_name not in h5_file:
                h5_file.create_group(group_name)

            for idx, frame in enumerate(frames):
                try:
                    face = self.cut_frame(frame, speaker)
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    faces = self.detector(gray, 1)

                    if len(faces) == 0:
                        logging.debug(f"No faces detected in frame {idx} for identifier: {identifier}")
                        continue

                    selected_face = None
                    if len(faces) > 1:
                        image_center = np.array((face.shape[1] // 2, face.shape[0] // 2))
                        distances = [
                            np.linalg.norm(
                                np.array(((face_rect.left() + face_rect.right()) // 2,
                                          (face_rect.top() + face_rect.bottom()) // 2)) - image_center)
                            for face_rect in faces
                        ]
                        selected_face = faces[np.argmin(distances)]
                    else:
                        selected_face = faces[0]

                    if selected_face:
                        x, y, w, h = selected_face.left(), selected_face.top(), selected_face.width(), selected_face.height()
                        cropped_face = face[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
                        cropped_face = cv2.resize(cropped_face, (self.face_size, self.face_size))

                        h5_file[group_name].create_dataset(str(idx), data=cropped_face, compression="gzip")

                except Exception as e:
                    logging.exception(f"Error processing frame {idx} for identifier {identifier}: {e}")

    def process_video_file(self, video, session_num, h5_file):
        """
        Process a single video file for a given session.

        Args:
            video (str): Name of the video file to process.
            session_num (int): Session number.
            h5_file (h5py.File): HDF5 file handle to save cropped face images.
        """
        video_path = f"{self.base_dir}/Session{session_num}/dialog/avi/DivX/{video}"

        map_path = f"{self.map_dir}/{video[:-4]}.csv"
        if not os.path.exists(map_path):
            logging.error(f"Map {map_path} does not exist, skipping video {video}.")
            return

        metadata = pd.read_csv(map_path)
        start_frames = metadata['start_frame'].tolist()
        end_frames = metadata['end_frame'].tolist()

        extracted_frames = self.extract_frames_from_ranges(video_path, start_frames, end_frames)
        if extracted_frames:
            self.detect_face(extracted_frames, metadata, h5_file)

    def process_session(self, n_session):
        """
        Process all videos in a session and save to HDF5 file.

        Args:
            n_session (int): Number of the session to process.
        """
        with h5py.File(self.face_h5_path, 'w') as h5_file:
            for session_num in range(1, n_session + 1):
                video_path = f"{self.base_dir}/Session{session_num}/dialog/avi/DivX"

                if not os.path.exists(video_path):
                    logging.error(f"Path {video_path} does not exist, skipping session {session_num}.")
                    continue

                videos = [file for file in os.listdir(video_path) if file.endswith('.avi') and not file.startswith('._')]

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(self.process_video_file, video, session_num, h5_file) for video in videos]
                    for future in futures:
                        future.result()


# Instantiate and start processing sessions
face_processor = FaceProcessor(
    face_h5_path=config.FACE_PATHH5,
    base_dir=config.BASE_PATH,
    map_dir=config.MAP_PATH,
    debug=config.DEBUG
)

face_processor.process_session(config.N_SESSIONS)

logging.info("Processing completed.")
