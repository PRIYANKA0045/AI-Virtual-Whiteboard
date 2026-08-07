import urllib.request
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. The categories we want our AI to learn
CLASSES = ['apple', 'car', 'cat', 'sun', 'tree']
SAMPLES_PER_CLASS = 10000  # We will use 10,000 drawings per category

def download_data():
    print("Downloading Quick, Draw! dataset...")
    os.makedirs("dataset", exist_ok=True)
    
    for label in CLASSES:
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{label}.npy"
        file_path = f"dataset/{label}.npy"
        if not os.path.exists(file_path):
            print(f"Downloading {label} drawings...")
            urllib.request.urlretrieve(url, file_path)
    print("Download complete!\n")

def load_data():
    x_data, y_data = [], []
    for idx, label in enumerate(CLASSES):
        # Load the drawings (they come as flattened 784-pixel arrays)
        data = np.load(f"dataset/{label}.npy")
        data = data[:SAMPLES_PER_CLASS] # Take a subset to train faster
        
        x_data.append(data)
        y_data.append(np.full(SAMPLES_PER_CLASS, idx))
        
    # Combine and reshape into 28x28 pixel images for the AI
    X = np.concatenate(x_data).astype('float32') / 255.0
    X = X.reshape(-1, 1, 28, 28)
    Y = np.concatenate(y_data)
    
    return torch.tensor(X), torch.tensor(Y, dtype=torch.long)

# 2. Build the Neural Network Architecture
class SketchCNN(nn.Module):
    def __init__(self, num_classes):
        super(SketchCNN, self).__init__()
        # Convolutional Layers to detect lines, curves, and shapes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Linear Layers to make the final guess
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    download_data()
    X, Y = load_data()
    
    # Prepare the data loader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = SketchCNN(num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting Training (This will take a few minutes)...")
    epochs = 5
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
    # 3. Save the trained brain!
    torch.save(model.state_dict(), "my_sketch_brain.pth")
    print("\nTraining Complete! Model saved as 'my_sketch_brain.pth'")

if __name__ == "__main__":
    train_model()