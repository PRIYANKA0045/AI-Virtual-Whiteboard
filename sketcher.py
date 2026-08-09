import cv2
import numpy as np
import torch
import torch.nn as nn

# These MUST match the categories you trained on!
CLASSES = ['apple', 'car', 'cat', 'sun', 'tree']

# We have to rebuild the exact same brain architecture so PyTorch knows where to put the saved weights
class SketchCNN(nn.Module):
    def __init__(self, num_classes):
        super(SketchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SketchAI:
    def __init__(self):
        print("Loading Custom Sketch AI Brain...")
        self.model = SketchCNN(num_classes=len(CLASSES))
        # Load your trained weights safely onto the CPU
        self.model.load_state_dict(torch.load("my_sketch_brain.pth", map_location=torch.device('cpu'), weights_only=True))
        self.model.eval()

    def guess_canvas(self, canvas_img):
        # 1. Grayscale & Threshold (White drawing, Black background)
        gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # 2. Find the drawing so we can crop the empty space around it
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "Nothing to guess!"
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 3. Crop with a little bit of padding so it looks like the training data
        padding = 40
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(thresh.shape[1], x + w + padding)
        y_end = min(thresh.shape[0], y + h + padding)
        
        cropped = thresh[y_start:y_end, x_start:x_end]
        
        if cropped.size == 0:
            return "Drawing too small!"
        
        # 4. Resize to 28x28 (The exact size our AI expects)
        resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 5. Convert to tensor and shape it (1 batch, 1 channel, 28 height, 28 width)
        img_array = np.array(resized, dtype=np.float32) / 255.0
        input_tensor = torch.tensor(img_array).view(1, 1, 28, 28)
        
        # 6. Make the prediction!
        with torch.no_grad():
            output = self.model(input_tensor)
            
        _, predicted_idx = torch.max(output, 1)
        guess = CLASSES[predicted_idx.item()]
        
        return f"Custom AI sees: A {guess.capitalize()}!"