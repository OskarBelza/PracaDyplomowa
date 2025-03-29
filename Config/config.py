# Description: Configuration file for the application

# Paths:
BASE_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_full_release/IEMOCAP_full_release'
MAP_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/EctractedData'
FACE_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/Faces_classes'
SPECTROGRAM_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/Spectograms_classes'
MULTIMODAL_DATA_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/MultimodalData'


# Number of sessions to process:
N_SESSIONS = 5

# Video frame rate:
VIDEO_RATE = 30

# Image size for face detection:
FACE_IMAGE_SIZE = 96
SPECTROGRAM_WIDTH = 24
SPECTROGRAM_HEIGHT = 128

# Values for face cropping:
Y1, Y2, Y3, Y4 = 130, 330, 130, 330
X1, X2, X3, X4 = 80, 280, 450, 650

# Audio processing values:
FFT_SIZE = 192
STEP_SIZE = 192 // 13.8
SPEC_THRESHOLD = 4

# Debug mode:
DEBUG = False

# Emotion labels:
CLASS_NAMES = ['Anger', 'Happiness', 'Excitement', 'Sadness', 'Frustration', 'Fear', 'Surprise', 'Neutral']

CLASS_SHORTCUTS_6 = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
CLASS_SHORTCUTS_10 = ['ang', 'dis', 'exc', 'fea', 'fru', 'hap', 'neu', 'oth', 'sad', 'sur']

# Batch size for training:
BATCH_SIZE = 32


