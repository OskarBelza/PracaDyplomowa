import os
import cv2
import dlib
import pandas as pd
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from Config import config

# Set up logging
logging.basicConfig(level=logging.DEBUG if config.DEBUG else logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global configuration
BASE_DIR = config.BASE_PATH
MAP_DIR = config.MAP_PATH
OUTPUT_DIR = config.FACE_PATH
FACE_SIZE = config.FACE_IMAGE_SIZE
Y1, Y2, X1, X2 = config.Y1, config.Y2, config.X1, config.X2
Y3, Y4, X3, X4 = config.Y3, config.Y4, config.X3, config.X4


def extract_frames_from_ranges(video_path, start_frames, end_frames):
    """
    Extract frames from a video based on given frame index ranges.

    This function loads the entire video into memory to avoid expensive frame-seeking
    operations. It then slices the preloaded list of frames based on the provided
    start and end frame indices.

    Args:
        video_path (str): Path to the video file.
        start_frames (list of int): List of start frame indices.
        end_frames (list of int): List of end frame indices.

    Returns:
        dict: A dictionary where each key is the index of the range, and the value
              is a list of frames within that range.
    """
    logging.info(f"Extracting frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return None

    # Read all frames into memory
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

    cap.release()
    logging.info(f"Total frames loaded: {len(all_frames)}")

    # Slice loaded frames based on the requested ranges
    extracted_frames = {}
    for i, (start, end) in enumerate(zip(start_frames, end_frames)):
        if end >= len(all_frames):
            end = len(all_frames) - 1
        if start >= len(all_frames):
            continue
        extracted_frames[i] = all_frames[start:end + 1]

    return extracted_frames


def convert_and_trim_bb(image, rect):
    """
    Convert a dlib rectangle object to a bounding box tuple (x, y, width, height),
    and ensure it stays within the image boundaries.

    Args:
        image (numpy.ndarray): The image in which the face was detected.
        rect (dlib.rectangle): The face detection result from dlib.

    Returns:
        tuple: Bounding box in (x, y, width, height) format.
    """
    startX = max(0, rect.left())
    startY = max(0, rect.top())
    endX = min(rect.right(), image.shape[1])
    endY = min(rect.bottom(), image.shape[0])
    return (startX, startY, endX - startX, endY - startY)


def process_video_file(video, session_num):
    """
    Process a single video file by extracting frame ranges, detecting faces,
    cropping them, and saving the results as individual image files categorized by emotion.

    Args:
        video (str): The filename of the video to process.
        session_num (int): The session number to determine the correct path.
    """
    try:
        # Load the CNN face detector model from dlib
        detector = dlib.cnn_face_detection_model_v1('../mmod_human_face_detector.dat')

        # Build paths to the video file and its corresponding map (CSV) file
        video_path = f"{BASE_DIR}/Session{session_num}/dialog/avi/DivX/{video}"
        map_path = f"{MAP_DIR}/{video[:-4]}.csv"

        if not os.path.exists(map_path):
            logging.error(f"Map {map_path} does not exist, skipping video {video}.")
            return

        metadata = pd.read_csv(map_path)
        start_frames = metadata['start_frame'].tolist()
        end_frames = metadata['end_frame'].tolist()

        # Extract the required frame ranges from the video
        extracted_frames = extract_frames_from_ranges(video_path, start_frames, end_frames)
        if not extracted_frames:
            return

        for dict_idx, frames in extracted_frames.items():
            speaker = metadata['speaker'][dict_idx]
            identifier = metadata['identifier'][dict_idx]
            emotion = metadata['emotion'][dict_idx]

            emotion_dir = os.path.join(OUTPUT_DIR, emotion)
            os.makedirs(emotion_dir, exist_ok=True)

            for idx, frame in enumerate(frames):
                # Crop the speaker's region from the frame
                face = frame[Y1:Y2, X1:X2] if speaker == 'L' else frame[Y3:Y4, X3:X4]

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                faces = detector(gray, 1)

                if len(faces) == 0:
                    logging.debug(f"Missing face in {idx} for {identifier}")
                    continue

                # Process each detected face bounding box
                boxes = [convert_and_trim_bb(face, r.rect) for r in faces]
                for (x, y, w, h) in boxes:
                    # Crop and resize the face image
                    cropped_face = cv2.resize(face[y:y + h, x:x + w], (FACE_SIZE, FACE_SIZE))

                    # Save the processed face image to file
                    filename = f"{identifier}_{idx}.png"
                    filepath = os.path.join(emotion_dir, filename)
                    cv2.imwrite(filepath, cropped_face)
                    logging.info(f"Saved in {filepath}")

    except Exception as e:
        logging.exception(f"Error loading file {video}: {e}")


def process_session(n_session):
    """
    Process all video files within the specified number of sessions.

    For each session, this function locates all video files in the expected directory,
    then distributes their processing across multiple worker processes using
    ProcessPoolExecutor. Each video is processed to detect and extract facial regions
    from selected frame ranges and save them as categorized face image files.

    Args:
        n_session (int): Total number of sessions to process.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for session_num in range(1, n_session + 1):
        video_path = f"{BASE_DIR}/Session{session_num}/dialog/avi/DivX"

        if not os.path.exists(video_path):
            logging.error(f"Directory {video_path} not found, skipping session {session_num}.")
            continue

        videos = [file for file in os.listdir(video_path) if file.endswith('.avi') and not file.startswith('._')]

        if not videos:
            logging.warning(f"No files found in {video_path}.")
            continue

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_video_file, video, session_num) for video in videos]
            for future in futures:
                future.result()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    process_session(config.N_SESSIONS)
    logging.info("Processing completed.")
