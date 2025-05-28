from tensorflow.keras import layers, models
from Config.config import NUM_CLASSES


def build_multimodal_model(audio_model, visual_model, num_classes=NUM_CLASSES):
    """
    Builds a multimodal emotion classification model that combines audio and visual features,
    using a multiplicative gating mechanism to weight each modality dynamically.

    Parameters:
        audio_model (tf.keras.Model): Pretrained or base model processing audio input.
        visual_model (tf.keras.Model): Pretrained or base model processing visual input (face images).
        num_classes (int): Number of emotion classes to classify.

    Returns:
        tf.keras.Model: Composite multimodal classification model.
    """

    # Inputs from unimodal models
    audio_input = audio_model.input
    visual_input = visual_model.input

    # Extracted features from both modalities
    audio_feat = audio_model.output
    visual_feat = visual_model.output

    # Concatenate features to compute modality gating weights
    concat_for_gating = layers.Concatenate()([audio_feat, visual_feat])

    # Dense layer that outputs two gating values (audio and video), with sigmoid activation
    gating_dense = layers.Dense(2, activation='sigmoid', name="modality_gate")(concat_for_gating)

    # Split the gating weights for each modality (shape: [batch_size, 1])
    gate_audio = layers.Lambda(lambda x: x[:, 0:1])(gating_dense)
    gate_visual = layers.Lambda(lambda x: x[:, 1:2])(gating_dense)

    # Apply gating weights to modality features (scale features)
    audio_weighted = layers.Multiply()([audio_feat, gate_audio])
    visual_weighted = layers.Multiply()([visual_feat, gate_visual])

    # Fuse gated features using addition
    fused = layers.Add()([audio_weighted, visual_weighted])

    # Dense transformation before classification
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.Dropout(0.4)(fused)

    # Output layer with softmax activation for emotion classification
    output = layers.Dense(num_classes, activation='softmax', name='emotion_output')(fused)

    # Assemble the full multimodal model
    model = models.Model(
        inputs=[audio_input, visual_input],
        outputs=output,
        name="MultimodalEmotionModelMultiplicative"
    )

    return model
