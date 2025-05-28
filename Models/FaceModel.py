from tensorflow.keras import models, layers
from Config.config import FACE_SIZE, NUM_CLASSES


def build_visual_model(frame_shape=(FACE_SIZE, FACE_SIZE, 3)):
    """
    Builds a base CNN model that processes a single face image frame.

    Parameters:
        frame_shape (tuple): Shape of the input image (height, width, channels).

    Returns:
        tf.keras.Model: CNN model that extracts features from face images, without classification layer.
    """
    # Input: face image frame
    inputs = layers.Input(shape=frame_shape, name='video_input')

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and dense layer (feature extractor)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Return the model as visual feature extractor
    return models.Model(inputs, x, name="VisualBranch")


def build_visual_classifier():
    """
    Builds a full CNN model for emotion classification based on face images.

    Returns:
        tf.keras.Model: Final classification model with softmax output over NUM_CLASSES classes.
    """
    # Load the base model
    base_model = build_visual_model()

    # Extract input and features
    inputs = base_model.input
    x = base_model.output

    # Output classification layer
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='visual_output')(x)

    # Return the complete classifier
    return models.Model(inputs, outputs, name="VisualClassifier")
