import matplotlib.pyplot as plt
import pandas as pd
import cv2

import Config.config


# Obsolete
def plot_class_distribution(csv_path, label_column, class_names=None):
    """
    Plots the distribution of classes from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        label_column (str): Name of the column containing class labels.
        class_names (list, optional): List of class names for better readability. If None, numeric labels are used.

    Returns:
        None
    """
    # Load the data
    data = pd.read_csv(csv_path)

    # Count the occurrences of each class
    class_counts = data[label_column].value_counts().sort_index()

    # Print the number of labels
    print("Number of labels in each class:")
    print(class_counts)

    # Map numeric labels to class names if provided
    if class_names:
        class_counts.index = [class_names[label] for label in class_counts.index]

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def show_sample_images(csv_path, label_column, image_column, class_names=None):
    """
    Displays a sample image for each class from the dataset.

    Args:
        csv_path (str): Path to the CSV file.
        label_column (str): Name of the column containing class labels.
        image_column (str): Name of the column containing image file paths.
        class_names (list, optional): List of class names for better readability. If None, numeric labels are used.

    Returns:
        None
    """
    # Load the data
    data = pd.read_csv(csv_path)

    # Get a sample image for each class
    unique_classes = data[label_column].unique()
    plt.figure(figsize=(15, 8))

    for i, class_label in enumerate(unique_classes):
        # Filter for the first image of the class
        sample_row = data[data[label_column] == class_label].iloc[0]
        image_path = sample_row[image_column]
        print(image_path)

        # Read and plot the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.subplot(1, len(unique_classes), i + 1)
        plt.imshow(image)
        plt.axis('off')

        # Add class name or numeric label as title
        title = class_names[class_label] if class_names else str(class_label)
        plt.title(title)

    plt.tight_layout()
    plt.show()


# Example usage
plot_class_distribution(Config.config.MULTIMODAL_DATA_PATH, 'emotion_label',
                        Config.config.CLASS_NAMES)
show_sample_images(Config.config.MULTIMODAL_DATA_PATH, 'emotion_label', 'face_path',
                   Config.config.CLASS_NAMES)
