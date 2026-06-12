import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import numpy as np
import os
import base64
import io
import cv2

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# DATA
# =========================
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

# =========================
# STRONG CNN MODEL
# =========================
class StrongCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = StrongCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAIN / LOAD MODEL
# =========================
def train_model():
    model.train()
    for epoch in range(6):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} done")

    torch.save(model.state_dict(), "model.pth")


if os.path.exists("model.pth"):
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("Model loaded")
    except:
        print("Model mismatch → retraining...")
        train_model()
else:
    print("Training new model...")
    train_model()

model.eval()

# =========================
# ADVANCED PREPROCESSING
# =========================
def preprocess_image(image):
    image = image.convert('L')
    image = ImageOps.invert(image)

    image = np.array(image)

    # threshold
    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # find bounding box
    coords = np.column_stack(np.where(image > 0))
    if coords.size == 0:
        return torch.zeros((1,1,28,28)).to(device)

    x, y, w, h = cv2.boundingRect(coords)
    digit = image[y:y+h, x:x+w]

    # resize
    digit = cv2.resize(digit, (20, 20))

    # pad
    padded = np.zeros((28, 28))
    padded[4:24, 4:24] = digit

    # normalize
    padded = padded / 255.0
    padded = (padded - 0.5) / 0.5

    tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor.to(device)

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']

    image_data = image_data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    img = preprocess_image(image)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))

        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [(int(i), float(probs[i])) for i in top3_idx]

    return jsonify({
        "prediction": pred,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "top3": top3
    })

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)