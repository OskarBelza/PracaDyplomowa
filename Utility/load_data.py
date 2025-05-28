from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from Config.config import FACE_SIZE, SPECTROGRAM_SIZE, BATCH_SIZE


def get_image_paths_and_labels(base_dir):
    """
    Retrieves image file paths and their corresponding labels from a directory structure:
    base_dir/class_name/image.png

    Returns:
        file_paths (List[str]): List of image paths.
        labels (List[int]): List of integer-encoded labels.
        class_names (List[str]): Sorted list of emotion class names.
    """
    file_paths, labels = [], []
    class_names = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(base_dir, class_name)
        for file_name in os.listdir(class_dir):
            path = os.path.join(class_dir, file_name)
            if os.path.isfile(path):
                file_paths.append(path)
                labels.append(label_idx)
    return file_paths, labels, class_names


def process(face_path, spec_path, label):
    """
    Loads and preprocesses a face image and its corresponding spectrogram.

    Returns:
        tuple: ((spectrogram_tensor, face_tensor), label)
    """
    face = tf.io.read_file(face_path)
    face = tf.image.decode_png(face, channels=3)
    face = tf.image.resize(face, (FACE_SIZE, FACE_SIZE))
    face = tf.cast(face, tf.float32) / 255.0

    spec = tf.io.read_file(spec_path)
    spec = tf.image.decode_png(spec, channels=3)
    spec = tf.image.resize(spec, (SPECTROGRAM_SIZE, SPECTROGRAM_SIZE))
    spec = tf.cast(spec, tf.float32) / 255.0

    return (spec, face), label


def build_dataset(face_paths, spec_paths, labels, shuffle=False):
    """
    Builds a tf.data.Dataset from lists of face image paths, spectrogram paths, and labels.

    Parameters:
        face_paths (List[str]): List of paths to face images.
        spec_paths (List[str]): List of paths to spectrogram images.
        labels (List[int]): List of corresponding integer labels.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: Batched and prefetched dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((face_paths, spec_paths, labels))
    dataset = dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(face_paths))
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def load_paired_dataset(face_dir, spec_dir, validation_split=0.15, test_split=0.15, seed=42):
    """
    Loads and splits a paired dataset of face images and spectrograms into train/val/test sets.

    Parameters:
        face_dir (str): Root directory with face images organized by class.
        spec_dir (str): Root directory with spectrograms, same structure as face_dir.
        validation_split (float): Proportion of validation data.
        test_split (float): Proportion of test data.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[tf.data.Dataset]: train_dataset, val_dataset, test_dataset
        List[str]: class_names
    """
    face_paths, labels, class_names = get_image_paths_and_labels(face_dir)
    spec_paths = [
        os.path.join(spec_dir, os.path.relpath(fp, face_dir)) for fp in face_paths
    ]

    face_temp, face_test, spec_temp, spec_test, labels_temp, labels_test = train_test_split(
        face_paths, spec_paths, labels, test_size=test_split, stratify=labels, random_state=seed
    )

    val_split_relative = validation_split / (1.0 - test_split)
    face_train, face_val, spec_train, spec_val, labels_train, labels_val = train_test_split(
        face_temp, spec_temp, labels_temp, test_size=val_split_relative,
        stratify=labels_temp, random_state=seed
    )

    train_dataset = build_dataset(face_train, spec_train, labels_train)
    val_dataset = build_dataset(face_val, spec_val, labels_val)
    test_dataset = build_dataset(face_test, spec_test, labels_test)

    return train_dataset, val_dataset, test_dataset, class_names


def load_dataset_with_explicit_test_split(
    data_dir,
    image_size=(128, 128),
    batch_size=32,
    seed=42,
    val_test_split=0.5
):
    """
    Loads image data from a directory using two steps:
    1. Loads with validation_split.
    2. Splits validation set into validation and test subsets.

    Parameters:
        data_dir (str): Root directory containing image data organized by class.
        image_size (Tuple[int, int]): Target image size.
        batch_size (int): Batch size.
        seed (int): Random seed for reproducibility.
        val_test_split (float): Proportion of validation data to use as validation (rest is test).

    Returns:
        Tuple[tf.data.Dataset]: train_ds, val_ds, test_ds
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    total_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_batches = int(total_batches * val_test_split)

    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    def normalize(x, y):
        return tf.cast(x, tf.float32) / 255.0, y

    train_ds = train_ds.map(normalize).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


