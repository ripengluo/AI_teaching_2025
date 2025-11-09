import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. Data Loading and Preprocessing
transform = transforms.Compose([
    transforms.Resize(96),                  # STL-10 has 96x96 images
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
])

# Load STL-10 dataset (10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)
train_dataset = datasets.STL10(
    root='./data',
    split='train',
    download=True,
    transform=transform
)
test_dataset = datasets.STL10(
    root='./data',
    split='test',
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Class names
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 
           'dog', 'horse', 'monkey', 'ship', 'truck']

# 2. Visualize Sample Images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# Get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images with labels
fig = plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    imshow(images[i])
    plt.title(classes[labels[i]])
plt.show()

# 3. Define CNN Model
class STL10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input: 3x96x96
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x96x96
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # 32x48x48
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64x48x48
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # 64x24x24
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128x24x24
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # 128x12x12
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128*12*12, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.classifier(x)
        return x

model = STL10_CNN()
print(model)

# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
def train(epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss/100:.3f}')
            running_loss = 0.0

# 6. Evaluation Function
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 7. Run Training and Evaluation
epochs = 50
best_acc = 0.0

for epoch in range(1, epochs+1):
    train(epoch)
    acc = test()
    
    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'stl10_cnn.pth')

print(f'Best Test Accuracy: {best_acc:.2f}%')

# 8. Visualize Predictions
def visualize_predictions():
    model.load_state_dict(torch.load('stl10_cnn.pth'))
    model.eval()
    
    # Get some test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    # Plot images with predictions
    fig = plt.figure(figsize=(12, 8))
    for idx in range(12):
        ax = fig.add_subplot(3, 4, idx+1, xticks=[], yticks=[])
        imshow(images[idx].cpu())
        ax.set_title(f'{classes[preds[idx]]} ({classes[labels[idx]]})',
                    color=('green' if preds[idx]==labels[idx] else 'red'))
    plt.show()

visualize_predictions()
