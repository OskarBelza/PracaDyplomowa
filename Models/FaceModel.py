from tensorflow.keras import models, layers
from Config.config import FACE_SIZE


def build_visual_model(frame_shape=(FACE_SIZE, FACE_SIZE, 3)):
    # Wejście: pojedyncza klatka
    inputs = layers.Input(shape=frame_shape, name='video_input')

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    return models.Model(inputs, x, name="VisualBranch")
