import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Data preprocessing: Convert to Tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),                      # Convert PIL Image or numpy array to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST
])

# Download and load training and test sets
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

# Display data shapes
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
print(f"Single image shape: {train_dataset[0][0].shape}")  # [C, H, W] = [1, 28, 28]

# Visualize sample images
fig, axes = plt.subplots(1, 6, figsize=(12, 2))
for i in range(6):
    img, label = train_dataset[i]
    axes[i].imshow(img.squeeze(), cmap='gray')  # Remove channel dim and display
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')
plt.show()

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel, 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer (2x2 window, stride 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer (64*7*7 → 128)
        self.fc2 = nn.Linear(128, 10)          # Output layer (128 → 10 classes)
        self.dropout = nn.Dropout(0.5)         # Prevent overfitting

    def forward(self, x):
        x = torch.relu(self.conv1(x))      # [B, 1, 28, 28] → [B, 32, 28, 28]
        x = self.pool(x)                   # [B, 32, 28, 28] → [B, 32, 14, 14]
        x = torch.relu(self.conv2(x))      # [B, 32, 14, 14] → [B, 64, 14, 14]
        x = self.pool(x)                   # [B, 64, 14, 14] → [B, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)         # Flatten [B, 64*7*7]
        x = torch.relu(self.fc1(x))        # [B, 64*7*7] → [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)                    # [B, 128] → [B, 10]
        return x

model = CNN()

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Train for 20 epochs
for epoch in range(20):
    train(epoch)

# Test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Evaluate model
test()

# Visualize predictions
model.eval()
with torch.no_grad():
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    pred = output.argmax(dim=1)

# Display first 12 predictions
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i in range(12):
    row, col = i // 4, i % 4
    axes[row, col].imshow(data[i].cpu().squeeze(), cmap='gray')
    axes[row, col].set_title(f"Pred: {pred[i].item()}, True: {target[i].item()}")
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()
