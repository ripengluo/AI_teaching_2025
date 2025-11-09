import torch
import torch.nn as nn
import torch.optim as optim
import numpy
import numpy as np
import matplotlib.pyplot as plt

# ========================================
# 1. Define a Deep Neural Network (DNN) Class
# ========================================
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Initialize a fully-connected DNN.
        Args:
            input_dim (int): Input feature dimension
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Output dimension
        """
        super(DNN, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# ========================================
# 2. Generate Synthetic Data
# ========================================
# True function: y = 2x1 + 0.5x2^2 + noise
X = torch.randn(1000, 2)  # 1000 samples, 2 features
y = 2*X[:, 0] + 0.5*X[:, 1]**2 + 0.1*torch.randn(1000)

# Train-test split
train_X, test_X = X[:800], X[800:]
train_y, test_y = y[:800], y[800:]


# ========================================
# 3. Initialize Model, Loss, and Optimizer
# ========================================
# Hyperparameters
input_dim = 2
hidden_dims = [64, 32]  # 2 hidden layers
output_dim = 1
learning_rate = 0.001
epochs = 100

# Initialize model
model = DNN(input_dim, hidden_dims, output_dim)

# Mean Squared Error (MSE) Loss
criterion = nn.MSELoss()  # Computes: 1/n Σ(y_pred - y_true)^2

# Adam optimizer with learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ========================================
# 4. Training Loop
# ========================================
model_save_path = "saved_model.pth"
for epoch in range(epochs):
    # Forward pass
    outputs = model(train_X).squeeze()
    loss = criterion(outputs, train_y)
    
    # Backward pass and optimize
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    # Print progress
    if (epoch+1) % 10 == 0:
        with torch.no_grad():  # Disable gradient tracking for evaluation
            test_preds = model(test_X).squeeze()
            test_loss = criterion(test_preds, test_y)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, model_save_path)


# ========================================
# 5. Evaluation
# ========================================
# Final test loss
with torch.no_grad():
    final_preds = model(test_X).squeeze()
    mse = criterion(final_preds, test_y)
    print(f"\nFinal Test MSE: {mse.item():.4f}")

def plot_diff():
    # Get predictions on test set
    with torch.no_grad():
        test_preds = model(test_X).squeeze()

    # Convert to numpy for plotting
    y_true = test_y.numpy()
    y_pred = test_preds.numpy()
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)  # Diagonal line
    plt.xlabel('Actual Values (y)', fontsize=12)
    plt.ylabel('Predicted Values (ŷ)', fontsize=12)
    plt.title('Actual vs Predicted Values', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Calculate and display R-squared
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    r_squared = corr**2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    return    

plot_diff()

