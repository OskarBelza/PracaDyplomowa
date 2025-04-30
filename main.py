from Models.SpectogramModel import build_audio_model
from Models.FaceModel import build_visual_model
from Models.MultiModalModel import build_multimodal_model
from Utility.load_data import load_paired_dataset
from Config.config import SPECTROGRAM_PATH, FACE_PATH, OUTPUT_PATH, BATCH_SIZE, EPOCHS
from Utility.evaluate import evaluate_multimodal_model


audio_model = build_audio_model()
visual_model = build_visual_model()
model = build_multimodal_model(audio_model, visual_model)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

train_dataset, val_dataset, test_dataset, class_names = load_paired_dataset(
    face_dir=FACE_PATH,
    spec_dir=SPECTROGRAM_PATH,
    validation_split=0.15,
    test_split=0.15,
)

model.fit(train_dataset, validation_data=val_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS)
model.save(f'{OUTPUT_PATH}/multimodal_model_multiplicative.keras')

evaluate_multimodal_model(model, test_dataset, class_names, output_path=f"{OUTPUT_PATH}/evaluation_report.txt")
