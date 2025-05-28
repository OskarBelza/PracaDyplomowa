from tensorflow.keras import models, layers
from Config.config import SPECTROGRAM_SIZE, NUM_CLASSES


def build_audio_model(input_shape=(SPECTROGRAM_SIZE, SPECTROGRAM_SIZE, 3)):
    """
    Builds a base CNN model that processes audio spectrograms as input images.
    The model consists of three Conv2D + MaxPooling blocks, followed by a dense layer with dropout.

    Parameters:
        input_shape (tuple): Shape of the input spectrogram images (default is RGB format).

    Returns:
        tf.keras.Model: Feature extraction model for the audio modality.
    """

    # Input: audio spectrogram represented as an image
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Return the model as a feature extractor
    return models.Model(inputs, x, name="AudioBranch")


def build_audio_classifier():
    """
    Builds a full audio-based emotion classification model.
    It uses the base feature extractor and adds a softmax classification head.

    Returns:
        tf.keras.Model: Complete classification model for audio inputs.
    """

    # Base feature extraction model
    base_model = build_audio_model()

    # Input and extracted features
    inputs = base_model.input
    x = base_model.output

    # Output layer classifying into NUM_CLASSES emotions
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='audio_output')(x)

    # Assemble the final model
    return models.Model(inputs, outputs, name="AudioClassifier")
