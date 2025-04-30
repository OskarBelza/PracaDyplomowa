import tensorflow as tf
from keras.config import enable_unsafe_deserialization
import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(class_distribution):
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
    plt.savefig('class_distribution.png', dpi=300)
    plt.show()


def load_and_describe_keras_model(model_path):
    # Włącz niebezpieczną deserializację (jeśli ufasz źródłu)
    enable_unsafe_deserialization()

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        return None

    print(f"\n✅ Model '{model_path}' został pomyślnie załadowany.\n")

    print("📐 Architektura modelu:")
    model.summary()

    print("\n🔍 Szczegóły warstw:")
    for i, layer in enumerate(model.layers):
        print(f"  {i+1}. Nazwa: {layer.name}")
        print(f"     Typ: {type(layer).__name__}")
        output_shape = getattr(layer, "output_shape", "Brak (np. InputLayer)")
        print(f"     Wyjście: {output_shape}")
        print(f"     Parametry: {layer.count_params()}")
        print("")

    print("⚙️ Parametry kompilacji:")
    if model.optimizer:
        print(f"  Optymalizator: {type(model.optimizer).__name__}")
        print(f"  Funkcja straty: {model.loss}")
        print(f"  Metryki: {model.metrics}")
    else:
        print("  Model nie został skompilowany.")

    return model


#model = load_and_describe_keras_model("../multimodal_model_multiplicative.keras")
