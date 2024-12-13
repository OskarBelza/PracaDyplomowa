import os
import cv2
import dlib
import numpy as np
import pandas as pd
import config


class FaceProcessor:
    def __init__(self, face_dir, video_dir, map_dir, y1=config.Y1, y2=config.Y2, x1=config.X1, x2=config.X2,
                 y3=config.Y3, y4=config.Y4, x3=config.X3, x4=config.X4, face_size=config.IMAGE_SIZE):
        """
        Initialize the FaceProcessor with directory paths and cropping coordinates.

        Args:
            face_dir (str): Directory to save cropped face images.
            video_dir (str): Directory containing video files.
            map_dir (str): Directory containing extraction maps.
            y1, y2, x1, x2, y3, y4, x3, x4: Coordinates for cropping frames based on speaker.
            face_size (int): Size of the cropped face images.
        """
        self.face_dir = face_dir
        self.video_dir = video_dir
        self.map_dir = map_dir
        self.y1, self.y2, self.x1, self.x2 = y1, y2, x1, x2
        self.y3, self.y4, self.x3, self.x4 = y3, y4, x3, x4
        self.face_size = face_size
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
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None

        extracted_frames = {}
        for i, (start, end) in enumerate(zip(start_frames, end_frames)):
            frames_in_range = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            for frame_idx in range(start, end + 1):
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read frame {frame_idx}.")
                    break
                frames_in_range.append(frame)

            extracted_frames[i] = frames_in_range

        cap.release()
        return extracted_frames

    def detect_face(self, frames_dict, metadata, face_dir):
        """
        Detect faces in video frames, crop them, and save to directory.

        Args:
            frames_dict (dict): Dictionary where keys are indices and values are lists of frames.
            metadata (pd.DataFrame): DataFrame containing speaker, identifier, and emotion metadata.
            face_dir (str): Directory to save cropped face images.
        """
        total_saved_images = 0

        for dict_idx, frames in frames_dict.items():
            speaker = metadata['speaker'][dict_idx]
            identifier = metadata['identifier'][dict_idx]
            emotion = metadata['emotion'][dict_idx]

            directory_path = f"{face_dir}/{identifier}_{emotion}"
            os.makedirs(directory_path, exist_ok=True)

            saved_images_count = 0

            for idx, frame in enumerate(frames):
                face = self.cut_frame(frame, speaker)
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 1)

                selected_face = None

                if len(faces) > 1:
                    image_center = np.array((face.shape[1] // 2, face.shape[0] // 2))
                    distances = []

                    for face_rect in faces:
                        face_center = np.array(
                            ((face_rect.left() + face_rect.right()) // 2,
                             (face_rect.top() + face_rect.bottom()) // 2)
                        )
                        distances.append(np.linalg.norm(image_center - face_center))

                    selected_face = faces[np.argmin(distances)]

                elif len(faces) == 1:
                    selected_face = faces[0]

                if selected_face:
                    x, y, w, h = selected_face.left(), selected_face.top(), selected_face.width(), selected_face.height()
                    cropped_face = face[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
                    cropped_face = cv2.resize(cropped_face, (self.face_size, self.face_size))

                    output_path = f"{directory_path}/{identifier}_{emotion}_{idx}.jpg"
                    cv2.imwrite(output_path, cropped_face)

                    saved_images_count += 1
                    total_saved_images += 1

            print(f"Saved {saved_images_count} images in directory: {directory_path}")

        print(f"Total saved images across all directories: {total_saved_images}")

    def process_session(self, session_num):
        """
        Process a single session of videos.

        Args:
            session_num (int): Session number to process.
        """
        video_path = f"{self.video_dir}/Session{session_num}/dialog/avi/DivX"

        if not os.path.exists(video_path):
            print(f"Path {video_path} does not exist, skipping session {session_num}.")
            return

        videos = [file for file in os.listdir(video_path) if file.endswith('.avi') and not file.startswith('._')]

        for video in videos:
            video_full_path = os.path.join(video_path, video)
            face_dir = f"{self.face_dir}/{video[:-4]}"
            os.makedirs(face_dir, exist_ok=True)

            map_path = f"{self.map_dir}/{video[:-4]}.csv"
            if not os.path.exists(map_path):
                print(f"Map {map_path} does not exist, skipping video {video}.")
                continue

            metadata = pd.read_csv(map_path)
            start_frames = metadata['first_frame'].tolist()
            end_frames = metadata['last_frame'].tolist()

            extracted_frames = self.extract_frames_from_ranges(video_full_path, start_frames, end_frames)
            if extracted_frames:
                self.detect_face(extracted_frames, metadata, face_dir)

            break


# Example usage
face_processor = FaceProcessor(
    face_dir=config.FACE_PATH,
    video_dir=config.BASE_PATH,
    map_dir=config.MAPA_PATH
)

for session in range(1, 2):  # Process only one session for simplicity
    face_processor.process_session(session)

print("Processing completed.")
