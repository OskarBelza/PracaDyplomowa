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
    Funkcja wizualizuje liczność klas emocji w zbiorze danych wejściowych oddzielnie dla modalności
    wizualnej (obrazy twarzy) i akustycznej (spektrogramy).

    Parametry:
        class_distribution (dict): słownik z licznością próbek dla każdej klasy osobno dla twarzy i dźwięku.
    """
    labels = list(class_distribution.keys())
    face_counts = [class_distribution[label]["face"] for label in labels]
    audio_counts = [class_distribution[label]["audio"] for label in labels]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x, face_counts, width=width, label='Obrazy twarzy')
    plt.bar([i + width for i in x], audio_counts, width=width, label='Spektrogramy audio')
    plt.xticks([i + width / 2 for i in x], labels, rotation=45)
    plt.ylabel("Liczba próbek")
    plt.title("Rozkład klas emocji (twarz vs. audio)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, output_path, labels=None, normalize='true'):
    """
    Funkcja generuje i wyświetla macierz pomyłek na podstawie etykiet rzeczywistych i predykcji modelu.

    Parametry:
        y_true (List[int]): rzeczywiste etykiety klas.
        y_pred (List[int]): przewidziane etykiety klas przez model.
        labels (List[str]): lista etykiet klas do podpisania osi.
        normalize (str): sposób normalizacji ('true' = normalizacja po wierszach).
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
