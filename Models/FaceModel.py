from tensorflow.keras import models, layers
from Config.config import FACE_SIZE, NUM_CLASSES

def build_visual_model(frame_shape=(FACE_SIZE, FACE_SIZE, 3)):
    """
    Buduje model bazowy przetwarzający pojedynczą klatkę obrazu twarzy.

    Parametry:
        frame_shape (tuple): Rozmiar wejściowego obrazu (wysokość, szerokość, kanały).

    Zwraca:
        tf.keras.Model: Model przetwarzający obrazy twarzy, bez warstwy klasyfikacyjnej.
    """
    # Wejście obrazu
    inputs = layers.Input(shape=frame_shape, name='video_input')

    # Blok konwolucyjny 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Blok konwolucyjny 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Blok konwolucyjny 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Warstwa spłaszczająca i gęsta (feature extractor)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Zwraca model bez klasyfikatora (VisualBranch)
    return models.Model(inputs, x, name="VisualBranch")


def build_visual_classifier():
    """
    Buduje pełny model klasyfikacyjny do rozpoznawania emocji na podstawie obrazu twarzy.

    Zwraca:
        tf.keras.Model: Model końcowy z warstwą softmax klasyfikującą do NUM_CLASSES klas.
    """
    # Pobierz model bazowy
    base_model = build_visual_model()

    # Wejście i wyjście z modelu bazowego
    inputs = base_model.input
    x = base_model.output

    # Warstwa wyjściowa klasyfikująca do NUM_CLASSES emocji
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='visual_output')(x)

    # Zwraca pełny model klasyfikacyjny
    return models.Model(inputs, outputs, name="VisualClassifier")
