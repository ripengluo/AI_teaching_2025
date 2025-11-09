#!/bin/python
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset (SMS Spam Collection)
data = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
data.columns = ["label", "text"]  # Rename columns

# Preview data
print(data.head())
print("\nSpam ratio:", data["label"].value_counts()["spam"] / len(data))

# Visualize class distribution
data["label"].value_counts().plot(kind="bar", color=["blue", "red"])
plt.title("Class Distribution (Ham vs. Spam)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# Convert labels to binary (0=ham, 1=spam)
y = (data["label"] == "spam").astype(int).values

# Extract TF-IDF features (limit to top 1000 words)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data["text"]).toarray()

print("Feature matrix shape:", X.shape)  # (5574 samples, 1000 features)

# Visualize top words in spam vs. ham
spam_words = vectorizer.get_feature_names_out()[X[y == 1].sum(axis=0).argsort()[-10:]]
ham_words = vectorizer.get_feature_names_out()[X[y == 0].sum(axis=0).argsort()[-10:]]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.barh(spam_words, X[y == 1][:, X[y == 1].sum(axis=0).argsort()[-10:]].sum(axis=0))
plt.title("Top 10 Spam Words")

plt.subplot(1, 2, 2)
plt.barh(ham_words, X[y == 0][:, X[y == 0].sum(axis=0).argsort()[-10:]].sum(axis=0))
plt.title("Top 10 Ham Words")
plt.tight_layout()
plt.show()


# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Define MLP model
class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Input layer → Hidden layer (64 neurons)
        self.fc2 = nn.Linear(64, 32)         # Hidden layer → Hidden layer
        self.fc3 = nn.Linear(32, 2)          # Output layer (2 classes: spam/ham)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation (CrossEntropyLoss includes Softmax)
        return x

# Initialize model, loss, and optimizer
model = SpamClassifier(X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 400
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()               # Clear gradients
    outputs = model(X_train)            # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    loss.backward()                     # Backpropagation
    optimizer.step()                    # Update weights
    losses.append(loss.item())          # Track loss

    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Plot training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()



# Test model
with torch.no_grad():
    test_outputs = model(X_test)
    predicted = torch.argmax(test_outputs, dim=1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy:.2%}")

# Confusion matrix
cm = confusion_matrix(y_test, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()



