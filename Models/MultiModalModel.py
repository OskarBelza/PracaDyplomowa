from tensorflow.keras import layers, models
from Config.config import NUM_CLASSES


def build_multimodal_model(audio_model, visual_model, num_classes=NUM_CLASSES):
    """
    Buduje multimodalny model do klasyfikacji emocji na podstawie sygnałów audio i wizualnych,
    wykorzystujący mechanizm multiplicative gating do ważenia modalności.

    Parametry:
        audio_model (tf.keras.Model): Wytrenowany lub bazowy model przetwarzający dane audio.
        visual_model (tf.keras.Model): Wytrenowany lub bazowy model przetwarzający dane wizualne (obrazy twarzy).
        num_classes (int): Liczba klas emocji do klasyfikacji.

    Zwraca:
        tf.keras.Model: Kompozytowy model multimodalny.
    """

    # Wejścia z modeli unimodalnych
    audio_input = audio_model.input
    visual_input = visual_model.input

    # Ekstrahowane cechy z obu modalności
    audio_feat = audio_model.output
    visual_feat = visual_model.output

    # Połączenie cech z obu źródeł do wyliczenia wag (bramek)
    concat_for_gating = layers.Concatenate()([audio_feat, visual_feat])

    # Warstwa gęsta generująca dwuwymiarowe bramki (dla audio i wideo), z aktywacją sigmoidalną
    gating_dense = layers.Dense(2, activation='sigmoid', name="modality_gate")(concat_for_gating)

    # Wydzielenie wag dla poszczególnych modalności (kształt: [batch_size, 1])
    gate_audio = layers.Lambda(lambda x: x[:, 0:1])(gating_dense)
    gate_visual = layers.Lambda(lambda x: x[:, 1:2])(gating_dense)

    # Zastosowanie wag do cech audio i wizualnych (skalowanie cech)
    audio_weighted = layers.Multiply()([audio_feat, gate_audio])
    visual_weighted = layers.Multiply()([visual_feat, gate_visual])

    # Fuzja zważonych cech przez dodawanie
    fused = layers.Add()([audio_weighted, visual_weighted])

    # Gęsta warstwa transformująca przestrzeń cech przed klasyfikacją
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.Dropout(0.4)(fused)

    # Warstwa wyjściowa klasyfikująca emocje (softmax)
    output = layers.Dense(num_classes, activation='softmax', name='emotion_output')(fused)

    # Złożenie modelu wejść i wyjść w jeden model multimodalny
    model = models.Model(
        inputs=[audio_input, visual_input],
        outputs=output,
        name="MultimodalEmotionModelMultiplicative"
    )

    return model
