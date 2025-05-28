from keras.config import enable_unsafe_deserialization
from Utility.load_data import load_paired_dataset
from Utility.evaluate import evaluate_multimodal_model
from Config.config import SPECTROGRAM_PATH, FACE_PATH, CLASS_DISTRIBUTION, OUTPUT_PATH
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)

enable_unsafe_deserialization()


def plot_class_distribution(class_distribution, output_path):
    """
    Visualizes the number of emotion class samples in the input dataset,
    separately for each modality: visual (face images) and acoustic (audio spectrograms).

    Parameters:
        class_distribution (dict): Dictionary with sample counts for each class,
                                   separately for face and audio modalities.
    """
    labels = list(class_distribution.keys())
    face_counts = [class_distribution[label]["face"] for label in labels]
    audio_counts = [class_distribution[label]["audio"] for label in labels]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x, face_counts, width=width, label='Face images')
    plt.bar([i + width for i in x], audio_counts, width=width, label='Audio spectrograms')
    plt.xticks([i + width / 2 for i in x], labels, rotation=45)
    plt.ylabel("Number of samples")
    plt.title("Emotion Class Distribution (Face vs. Audio)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, output_path, labels=None, normalize='true'):
    """
    Generates and displays a confusion matrix based on true labels and model predictions.

    Parameters:
        y_true (List[int]): Ground truth class labels.
        y_pred (List[int]): Predicted class labels from the model.
        labels (List[str]): Optional list of class names for axis labeling.
        normalize (str): Normalization method ('true' = normalize per row).
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(output_path, dpi=300)
    plt.show()



"""
#model = tf.keras.models.load_model("../Outputs/multimodal_model_multiplicative.keras")


train_dataset, val_dataset, test_dataset, class_names = load_paired_dataset(
    face_dir=FACE_PATH,
    spec_dir=SPECTROGRAM_PATH,
    validation_split=0.15,
    test_split=0.15,
)


#y_true, y_pred = evaluate_multimodal_model(model, test_dataset, class_names, output_path="../Outputs/evaluation_report.txt")

#plot_confusion_matrix(y_true, y_pred,output_path=f"{OUTPUT_PATH}/confusion_matrix.png", labels=class_names, normalize='true')
#plot_class_distribution(CLASS_DISTRIBUTION)
"""

# Uncomment the above lines to run the evaluation and plotting functions"""
