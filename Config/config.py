# Description: Configuration file for the application

# Paths:
BASE_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_full_release/IEMOCAP_full_release'
MAP_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/EctractedData'
FACE_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/Faces_classes'
SPECTROGRAM_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/Spectograms_classes'
OUTPUT_PATH = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/Outputs'

# Number of sessions to process:
N_SESSIONS = 5

# Video frame rate:
VIDEO_RATE = 30

# Image size for face detection:
FACE_SIZE = 96
SPECTROGRAM_SIZE = 128

# Values for face cropping:
Y1, Y2, Y3, Y4 = 130, 330, 130, 330
X1, X2, X3, X4 = 80, 280, 450, 650

# Audio processing values:
FFT_SIZE = 512
HOP_LENGTH = 128
MELS = 128
FMIN = 0


# Debug mode:
DEBUG = False

# Emotion labels:
CLASS_NAMES = ['Anger', 'Happiness', 'Excitement', 'Sadness', 'Frustration', 'Fear', 'Surprise', 'Neutral']

CLASS_SHORTCUTS_6 = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
CLASS_SHORTCUTS_10 = ['ang', 'dis', 'exc', 'fea', 'fru', 'hap', 'neu', 'oth', 'sad', 'sur']
NUM_CLASSES = 6

# Batch size for training:
BATCH_SIZE = 32
EPOCHS = 20

MIN_AUDIO_LENGTH = 0.1

CLASS_DISTRIBUTION = {
    "neutralność (neu)": {"face": 147690, "audio": 151429},
    "frustracja (fru)": {"face": 189531, "audio": 196591},
    "obrzydzenie (dis)": {"face": 97, "audio": 97},
    "zaskoczenie (sur)": {"face": 6458, "audio": 6701},
    "złość (ang)": {"face": 102701, "audio": 103571},
    "ekscytacja (exc)": {"face": 104273, "audio": 113740},
    "inne (oth)": {"face": 415, "audio": 427},
    "smutek (sad)": {"face": 157707, "audio": 163057},
    "strach (fea)": {"face": 2285, "audio": 2542},
    "radość (hap)": {"face": 55156, "audio": 58373}
}
