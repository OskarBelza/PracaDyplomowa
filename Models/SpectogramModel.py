from tensorflow.keras import models, layers
from Config.config import SPECTROGRAM_SIZE, NUM_CLASSES


def build_audio_model(input_shape=(SPECTROGRAM_SIZE, SPECTROGRAM_SIZE, 3)):
    """
    Buduje bazowy model CNN przetwarzający dane audio w postaci spektrogramów.

    Parametry:
        input_shape (tuple): Wymiary wejściowe obrazu spektrogramu (domyślnie RGB).

    Zwraca:
        tf.keras.Model: Model przetwarzający dane audio i zwracający wektor cech.
    """

    # Warstwa wejściowa
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # Pierwsza warstwa konwolucyjna + pooling
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Druga warstwa konwolucyjna + pooling
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Trzecia warstwa konwolucyjna + pooling
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Spłaszczenie cech i warstwa gęsta z dropoutem
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Zwraca model wyjściowy bez klasyfikatora
    return models.Model(inputs, x, name="AudioBranch")


def build_audio_classifier():
    """
    Buduje kompletny model klasyfikacji audio, bazując na wyjściu z modelu cech.

    Zwraca:
        tf.keras.Model: Model końcowy klasyfikujący spektrogramy na klasy emocji.
    """

    # Budowa bazowego modelu ekstrakcji cech
    base_model = build_audio_model()

    # Podłączenie klasyfikatora do wyjścia bazowego modelu
    inputs = base_model.input
    x = base_model.output
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='audio_output')(x)

    # Złożenie całościowego modelu
    return models.Model(inputs, outputs, name="AudioClassifier")
