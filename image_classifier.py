import os
import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "model/image_model.onnx"
DATA_DIR = "generated_ids/"

class IDCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['genuine', 'fake', 'suspicious']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        # Load all images and labels
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and transform image
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        
        return img, label

class IDCardClassifier(nn.Module):
    def __init__(self):
        super(IDCardClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_and_save_model():
    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = IDCardDataset(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDCardClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    print(f"Training on {device}")
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%\n')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the model in ONNX format
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=device)
            torch.onnx.export(model, 
                            dummy_input, 
                            MODEL_PATH,
                            opset_version=13,
                            input_names=['input'],
                            output_names=['output'],
                            dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
            print(f"New best model saved in ONNX format to {MODEL_PATH}")

def load_trained_model():
    return ort.InferenceSession(MODEL_PATH)

def classify_image(pil_image, model, class_names):
    """
    Args:
        pil_image: PIL.Image.Image - input image object
        model: ONNX InferenceSession
        class_names: list of class names (in order)
    Returns:
        label: predicted class label (str)
        confidence: prediction confidence (float)
    """
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(pil_image).unsqueeze(0).numpy()
    
    # Run inference
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    predictions = model.run([output_name], {input_name: img_tensor.astype(np.float32)})[0]
    
    # Get prediction and confidence
    predicted_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_idx])
    label = class_names[predicted_idx]
    
    return label, confidence
if __name__ == "__main__":
    print("Starting training...")
    train_and_save_model()
