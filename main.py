import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from Models.MultiModalModel import MultiModalModel


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            face_inputs, spec_inputs, labels = batch
            face_inputs, spec_inputs, labels = face_inputs.to(device), spec_inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(face_inputs, spec_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")


# Example usage
if __name__ == "__main__":
    # Create synthetic dataset
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples
            self.face_data = torch.randn(num_samples, 3, 112, 112)  # Face images
            self.spec_data = torch.randn(num_samples, 1, 112, 112)  # Spectrograms
            self.labels = torch.randint(0, 8, (num_samples,))

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.face_data[idx], self.spec_data[idx], self.labels[idx]

    # Initialize model, dataset, dataloader, criterion, and optimizer
    model = MultiModalModel(face_embedding_size=128, spec_embedding_size=128, num_classes=8)
    dataset = SyntheticDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)
