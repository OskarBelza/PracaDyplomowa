# Description: Configuration file for the application

# Paths:
BASE_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_full_release/IEMOCAP_full_release'
MAP_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/EctractedData'
FACE_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/Faces'
SPECTROGRAM_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/Spectograms'
MULTIMODAL_DATA_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/MultimodalData'


# Number of sessions to process:
N_SESSIONS = 5

# Video frame rate:
VIDEO_RATE = 30

# Image size for face detection:
IMAGE_SIZE = 48

# Values for face cropping:
Y1, Y2, Y3, Y4 = 130, 330, 130, 330
X1, X2, X3, X4 = 80, 280, 450, 650

# Audio processing values:
FFT_SIZE = 192
STEP_SIZE = 192 // 13.8
SPEC_THRESHOLD = 4

# Debug mode:
DEBUG = False
