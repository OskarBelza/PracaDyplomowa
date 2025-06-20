# Multimodal Emotion Recognition from Audio and Facial Images (IEMOCAP)

This project implements a complete multimodal deep learning pipeline for emotion recognition using facial images and audio signals. It leverages the IEMOCAP dataset and combines CNN-based feature extractors with a multiplicative gating mechanism to fuse audio-visual features.

---

## 📁 Project Structure

```
├── .venv/                         # Virtual environment (not tracked)
├── Config/
│   └── config.py                  # Global configuration parameters
├── Models/
│   ├── FaceModel.py               # CNN model for face images
│   ├── MultiModalModel.py         # Fusion model with multiplicative gating
│   └── SpectogramModel.py         # CNN model for spectrograms
├── Outputs/                       # Trained models and evaluation outputs (not tracked)
├── data/                          # IEMOCAP dataset directory (not tracked)  
├── Preprocessing/
│   ├── AudioProcessor.py          # Converts audio to mel-spectrograms
│   ├── FaceProcessor.py           # Extracts face regions from video frames
│   └── MapProcessor.py            # Parses annotation files to CSV
├── Utility/
│   ├── clean_unpaired_data.py     # Removes unpaired face/audio samples
│   ├── count.py                   # Counts per-class samples
│   ├── evaluate.py                # Model evaluation tools
│   ├── load_data.py               # Loads and splits dataset
│   └── visualization.py           # Class distribution and confusion matrix
├── .gitignore                     # Git ignore file
├── main.py                        # Training and evaluation entry point
├── mmod_human_face_detector.dat   # dlib's pretrained face detector 
├── README.md                      # Project description and instructions
└── requirements.txt               # Required Python packages
```

---

## 🧠 Model Description

The multimodal model consists of two unimodal branches:

* **AudioBranch**: CNN that processes mel-spectrograms.
* **VisualBranch**: CNN that processes cropped face images.

Features from both branches are combined using a learned **multiplicative gating** mechanism. Gated features are fused and passed through dense layers for emotion classification.

See: `Models/MultiModalModel.py`

---

## 🔄 Data Processing Pipeline

1. **Annotation Mapping** – `Preprocessing/MapProcessor.py` converts IEMOCAP `.txt` files into structured `.csv` metadata.
2. **Face Extraction** – `Preprocessing/FaceProcessor.py` uses dlib's CNN detector to crop faces from video frames.
3. **Audio Processing** – `Preprocessing/AudioProcessor.py` generates mel-spectrograms from audio segments.
4. **Data Cleaning** – `Utility/clean_unpaired_data.py` ensures only paired (face + audio) samples remain.

---

## 📊 Dataset Format

After processing, the data is organized by emotion class:

```
Outputs/
├── Faces/
│   ├── happy/
│   ├── angry/
│   └── ...
└── Spectrograms/
    ├── happy/
    ├── angry/
    └── ...
```

Sample statistics and visualizations are available in `Utility/visualization.py` and `Utility/count.py`.

---

## ⚙️ Training

The `main.py` script compiles the multimodal model, trains it using paired spectrogram-face data, saves the final model, and evaluates performance:

```bash
python main.py
```

---

## 🧪 Evaluation

The evaluation report includes:

* Balanced Accuracy
* Full classification report (precision, recall, f1-score)
* Confusion matrix (optional normalization)

Evaluation tools and visualization available in `Utility/evaluate.py` and `Utility/visualization.py`.

---

## ✅ Requirements

* Python
* TensorFlow
* NumPy, OpenCV, SciPy, pandas, scikit-learn
* dlib (with `mmod_human_face_detector.dat`)
* librosa

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📘 Example Usage

```bash
# 1. Convert annotations
python Preprocessing/MapProcessor.py

# 2. Process video and audio
python Preprocessing/FaceProcessor.py
python Preprocessing/AudioProcessor.py

# 3. Clean unpaired samples
python Utility/clean_unpaired_data.py

# 4. Train and evaluate the model
python main.py
```

---

## ✍️ Authors

This project was developed as part of a bachelor's thesis by Oskar Bełza on multimodal emotion recognition using the IEMOCAP dataset.
