from tensorflow.keras import layers, models
from Config.config import NUM_CLASSES


def build_multimodal_model(audio_model, visual_model, num_classes=NUM_CLASSES):
    audio_input = audio_model.input
    visual_input = visual_model.input

    audio_feat = audio_model.output
    visual_feat = visual_model.output

    concat_for_gating = layers.Concatenate()([audio_feat, visual_feat])

    gating_dense = layers.Dense(2, activation='sigmoid', name="modality_gate")(concat_for_gating)

    gate_audio = layers.Lambda(lambda x: x[:, 0:1])(gating_dense)
    gate_visual = layers.Lambda(lambda x: x[:, 1:2])(gating_dense)

    audio_weighted = layers.Multiply()([audio_feat, gate_audio])
    visual_weighted = layers.Multiply()([visual_feat, gate_visual])

    fused = layers.Add()([audio_weighted, visual_weighted])
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.Dropout(0.4)(fused)

    output = layers.Dense(num_classes, activation='softmax', name='emotion_output')(fused)

    model = models.Model(inputs=[audio_input, visual_input], outputs=output, name="MultimodalEmotionModelMultiplicative")
    return model
