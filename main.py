from Models.FaceModel import create_face_model
from Models.SpectogramModel import create_spectrogram_model
from Utility.load_data import load_spectrogram_dataset, load_face_dataset
from tensorflow.keras import layers
from sklearn.metrics import classification_report, balanced_accuracy_score
import cv2 as cv
from Utility.visualization import plot_spectrograms_from_tf_dataset
from Config.config import CLASS_NAMES



# Tworzenie modeli
#face_model = create_face_model()
spectrogram_model = create_spectrogram_model()

# Podsumowanie
#face_model.summary()
spectrogram_model.summary()


# Wczytanie danych
#face_train_ds, face_val_ds, face_classes = load_face_dataset(batch_size=128)
spec_train_ds, spec_val_ds, spec_classes = load_spectrogram_dataset(batch_size=128)

# Wizualizacja
plot_spectrograms_from_tf_dataset(spec_train_ds, CLASS_NAMES, num_images=12)

# Trenowanie modeli
#history = face_model.fit(face_train_ds, validation_data=face_val_ds, epochs=10)
spectrogram_model.fit(spec_train_ds, validation_data=spec_val_ds, epochs=10)

# Zapisać model
#face_model.save('face_model.keras')
spectrogram_model.save('spectrogram_model.h5')

#report = classification_report(face_val_ds, face_model.predict(face_val_ds), )
#print(report)

