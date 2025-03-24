import tensorflow as tf
import os
from Config.config import SPECTROGRAM_PATH, FACE_PATH


def load_tiny_face_dataset(face_dir=FACE_PATH, img_size=(48, 48), batch_size=64,
                           validation_split=0.2, seed=42, subset_percent=0.01):
    """
    Loads a small fraction (e.g. 1%) of face dataset for testing purposes.

    Args:
        face_dir (str): Directory containing face images categorized by emotion.
        img_size (tuple): Target image size.
        batch_size (int): Batch size.
        validation_split (float): Fraction of data used for validation.
        seed (int): Seed for shuffling.
        subset_percent (float): Fraction of the dataset to use (e.g. 0.01 = 1%).

    Returns:
        train_ds (tf.data.Dataset), val_ds (tf.data.Dataset), class_names (list)
    """
    class_names = sorted(os.listdir(face_dir))

    # Load full training and validation datasets
    train_full = tf.keras.utils.image_dataset_from_directory(
        face_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        validation_split=validation_split,
        subset="training",
        seed=seed
    )

    val_full = tf.keras.utils.image_dataset_from_directory(
        face_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed
    )

    # Oblicz liczbę batchy do pobrania (1% danych)
    num_train_batches = tf.data.experimental.cardinality(train_full).numpy()
    num_val_batches = tf.data.experimental.cardinality(val_full).numpy()

    take_train = max(1, int(num_train_batches * subset_percent))
    take_val = max(1, int(num_val_batches * subset_percent))

    train_ds = train_full.take(take_train)
    val_ds = val_full.take(take_val)

    # Normalizacja
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def load_tiny_spectrogram_dataset(spec_dir=SPECTROGRAM_PATH, img_size=(48, 48), batch_size=64,
                                   validation_split=0.2, seed=42, subset_percent=0.01):
    """
    Loads a small fraction (e.g. 1%) of spectrogram dataset for testing purposes.

    Args:
        spec_dir (str): Directory containing spectrogram images categorized by emotion.
        img_size (tuple): Target image size.
        batch_size (int): Batch size.
        validation_split (float): Fraction of data used for validation.
        seed (int): Seed for shuffling.
        subset_percent (float): Fraction of the dataset to use (e.g. 0.01 = 1%).

    Returns:
        train_ds (tf.data.Dataset), val_ds (tf.data.Dataset), class_names (list)
    """
    class_names = sorted(os.listdir(spec_dir))

    # Load full training and validation datasets
    train_full = tf.keras.utils.image_dataset_from_directory(
        spec_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        validation_split=validation_split,
        subset="training",
        seed=seed
    )

    val_full = tf.keras.utils.image_dataset_from_directory(
        spec_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed
    )

    # Oblicz liczbę batchy do pobrania (1% danych)
    num_train_batches = tf.data.experimental.cardinality(train_full).numpy()
    num_val_batches = tf.data.experimental.cardinality(val_full).numpy()

    take_train = max(1, int(num_train_batches * subset_percent))
    take_val = max(1, int(num_val_batches * subset_percent))

    train_ds = train_full.take(take_train)
    val_ds = val_full.take(take_val)

    # Normalizacja
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


face_train_ds, face_val_ds, face_classes = load_tiny_face_dataset(subset_percent=0.01)
spec_train_ds, spec_val_ds, spec_classes = load_tiny_spectrogram_dataset(subset_percent=0.01)
print(len(face_train_ds), len(face_val_ds), face_classes)
print(len(spec_train_ds), len(spec_val_ds), spec_classes)
