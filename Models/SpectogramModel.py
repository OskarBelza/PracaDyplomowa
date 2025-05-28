from tensorflow.keras import models, layers
from Config.config import SPECTROGRAM_SIZE, NUM_CLASSES


def build_audio_model(input_shape=(SPECTROGRAM_SIZE, SPECTROGRAM_SIZE, 3)):
    """
    Buduje model bazowy przetwarzający spektrogramy audio jako obrazy wejściowe.
    Model składa się z trzech bloków Conv2D + MaxPooling, a następnie warstwy gęstej z dropoutem.

    Parametry:
        input_shape (tuple): Rozmiar wejściowych spektrogramów (domyślnie RGB).

    Zwraca:
        tf.keras.Model: Model ekstrakcji cech z modalności audio.
    """

    # Wejście: spektrogram audio w postaci obrazu
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # Blok konwolucyjny 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Blok konwolucyjny 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Blok konwolucyjny 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Spłaszczenie i warstwa w pełni połączona
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Zwracamy model jako ekstraktor cech z danych audio
    return models.Model(inputs, x, name="AudioBranch")


def build_audio_classifier():
    """
    Tworzy pełny model klasyfikacji emocji na podstawie danych audio.
    Wykorzystuje bazowy model ekstrakcji cech i dodaje końcową warstwę softmax.

    Zwraca:
        tf.keras.Model: Model klasyfikacyjny dla danych akustycznych.
    """

    # Bazowy model do ekstrakcji cech
    base_model = build_audio_model()

    # Wejście i wyjście z modelu bazowego
    inputs = base_model.input
    x = base_model.output

    # Warstwa wyjściowa klasyfikująca na 6 emocji
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='audio_output')(x)

    # Złożenie końcowego modelu
    return models.Model(inputs, outputs, name="AudioClassifier")
