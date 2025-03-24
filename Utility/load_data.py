import tensorflow as tf
import os
from Config.config import FACE_PATH, SPECTROGRAM_PATH


def load_face_dataset(face_dir=FACE_PATH, img_size=(48, 48), batch_size=64,
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
    # Get class names (subdirectories)
    class_names = sorted(os.listdir(face_dir))

    # Load dataset with `image_dataset_from_directory`
    train_ds = tf.keras.utils.image_dataset_from_directory(
        face_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
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
        validation_split=validation_split,
        subset="validation",
        seed=seed
    )

    # Normalize to [0, 1] range
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch to improve performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def load_spectrogram_dataset(spec_dir=SPECTROGRAM_PATH, img_size=(48, 48), batch_size=64,
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
    # Get class names (subdirectories)
    class_names = sorted(os.listdir(spec_dir))

    train_ds = tf.keras.utils.image_dataset_from_directory(
        spec_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
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
        validation_split=validation_split,
        subset="validation",
        seed=seed
    )

    # Normalize to [0, 1]
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


#face_train_ds, face_val_ds, face_classes = load_face_dataset(batch_size=128)
spec_train_ds, spec_val_ds, spec_classes = load_spectrogram_dataset(batch_size=128)

