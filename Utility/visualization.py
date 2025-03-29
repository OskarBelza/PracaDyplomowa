import matplotlib.pyplot as plt
import numpy as np


def plot_spectrograms_from_tf_dataset(dataset, class_names=None, num_images=9):
    """
    Wizualizuje spektrogramy z tf.data.Dataset.

    Args:
        dataset (tf.data.Dataset): Dataset zwracający (image, label).
        class_names (list): Nazwy klas (np. ['ang', 'exc', ...]).
        num_images (int): Liczba obrazów do wyświetlenia.
    """
    images = []
    labels = []

    # Zbierz obrazy i etykiety z pierwszych kilku batchy
    for batch_images, batch_labels in dataset.take(1):
        images.extend(batch_images.numpy())
        labels.extend(batch_labels.numpy())

    # Przytnij do num_images
    images = images[:num_images]
    labels = labels[:num_images]

    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.astype("uint8"))  # już wczytane jako RGB i uint8
        if class_names:
            plt.title(class_names[label], fontsize=10)
        else:
            plt.title(f"Label: {label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
