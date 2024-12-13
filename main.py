import os, os.path, time
import librosa
import numpy as np
import cv2
import pandas as pd

n_sessions = 5
video_rate = 30
first_line = 2

if not os.path.exists('IEMOCAP_full_release/IEMOCAP_data'):
    os.mkdir('IEMOCAP_full_release/IEMOCAP_data')

for ses in range(1, n_sessions+1):

    EMOEVALUATION_PATH = 'IEMOCAP_full_release/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(ses)
    IEMOCAP_PATH = 'IEMOCAP_full_release/IEMOCAP_full_release/Session{}/dialog/'.format(ses)

    evaluations = []

    # Wybieramy pliki z zapisanymi ewaluacjami
    for file in os.listdir(EMOEVALUATION_PATH):
        if file.endswith('.txt'):
            evaluations.append(file)

    for eval in evaluations:
        # Odczytujemy plik z ewaluacja
        extract_values()