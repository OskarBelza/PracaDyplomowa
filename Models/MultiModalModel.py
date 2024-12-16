import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.SpectogramModel import SpectrogramCNN
from Models.FaceModel import FaceCNN


class MultiModalModel(nn.Module):
    def __init__(self, face_embedding_size=128, spec_embedding_size=128, num_classes=10):
        super(MultiModalModel, self).__init__()
        self.face_cnn = SpectrogramCNN()
        self.spec_cnn = FaceCNN(embedding_size=spec_embedding_size)

        # Fully connected layers for combined embeddings
        self.fc1 = nn.Linear(face_embedding_size + spec_embedding_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, face_input, spec_input):
        face_embedding = self.face_cnn(face_input)
        spec_embedding = self.spec_cnn(spec_input)

        # Concatenate embeddings
        combined = torch.cat((face_embedding, spec_embedding), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
