import tensorflow as tf
from Config.config import (FACE_PATH, SPECTROGRAM_PATH, CLASS_SHORTCUTS_10, CLASS_SHORTCUTS_6, SPECTROGRAM_HEIGHT,
                           SPECTROGRAM_WIDTH, FACE_IMAGE_SIZE)
from tensorflow.keras import layers


def preprocess(train_ds, val_ds):
    normalization_layer = layers.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds


def load_face_dataset(face_dir=FACE_PATH, img_size=(FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), batch_size=64,
                      validation_split=0.2, seed=42):
    """
    Loads face images into TensorFlow datasets, splitting into training and validation sets.

    Args:
        face_dir (str): Directory containing face images categorized by emotion.
        img_size (tuple): Target image size (width, height).
        batch_size (int): Batch size for training.
        validation_split (float): Percentage of data for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        class_names (list): List of emotion class labels.
    """

    class_names = CLASS_SHORTCUTS_6

    # Load dataset with `image_dataset_from_directory`
    train_ds = tf.keras.utils.image_dataset_from_directory(
        face_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        class_names=class_names,
        validation_split=validation_split,
        subset="training",
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        face_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        class_names=class_names,
        validation_split=validation_split,
        subset="validation",
        seed=seed
    )

    train_ds, val_ds = preprocess(train_ds, val_ds)
    return train_ds, val_ds, class_names


def load_spectrogram_dataset(spec_dir=SPECTROGRAM_PATH, img_size=(SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH), batch_size=64,
                              validation_split=0.2, seed=42):
    """
    Loads spectrogram images into TensorFlow datasets, splitting into training and validation sets.

    Args:
        spec_dir (str): Directory containing spectrogram images categorized by emotion.
        img_size (tuple): Target image size (width, height).
        batch_size (int): Batch size for training.
        validation_split (float): Percentage of data for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        class_names (list): List of emotion class labels.
    """

    class_names = CLASS_SHORTCUTS_6

    train_ds = tf.keras.utils.image_dataset_from_directory(
        spec_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        class_names=class_names,
        validation_split=validation_split,
        subset="training",
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        spec_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        class_names=class_names,
        validation_split=validation_split,
        subset="validation",
        seed=seed
    )

    #train_ds, val_ds = preprocess(train_ds, val_ds)
    return train_ds, val_ds, class_names


#face_train_ds, face_val_ds, face_classes = load_face_dataset(batch_size=128)
#spec_train_ds, spec_val_ds, spec_classes = load_spectrogram_dataset(batch_size=128)

