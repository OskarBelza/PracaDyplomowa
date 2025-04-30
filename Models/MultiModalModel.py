from tensorflow.keras import layers, models
from Config.config import NUM_CLASSES


def build_multimodal_model(audio_model, visual_model, num_classes=NUM_CLASSES):
    # Wejścia
    audio_input = audio_model.input
    visual_input = visual_model.input

    # Wyjścia strumieni
    audio_feat = audio_model.output  # (None, 128)
    visual_feat = visual_model.output  # (None, 128)

    # Konkatenacja cech dla obliczenia wag (ale bez softmax)
    concat_for_gating = layers.Concatenate()([audio_feat, visual_feat])

    # Multiplicative gating: osobne skalarne wagi dla audio i visual
    gating_dense = layers.Dense(2, activation='sigmoid', name="modality_gate")(concat_for_gating)

    gate_audio = layers.Lambda(lambda x: x[:, 0:1])(gating_dense)
    gate_visual = layers.Lambda(lambda x: x[:, 1:2])(gating_dense)

    # Wektorowe ważenie modalności
    audio_weighted = layers.Multiply()([audio_feat, gate_audio])
    visual_weighted = layers.Multiply()([visual_feat, gate_visual])

    # Sumowanie cech
    fused = layers.Add()([audio_weighted, visual_weighted])
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.Dropout(0.4)(fused)

    # Wyjście klasyfikacyjne
    output = layers.Dense(num_classes, activation='softmax', name='emotion_output')(fused)

    # Finalny model
    model = models.Model(inputs=[audio_input, visual_input], outputs=output, name="MultimodalEmotionModelMultiplicative")
    return model