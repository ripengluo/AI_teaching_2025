import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected  # import missing utility
from sklearn.metrics import accuracy_score
import numpy as np

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# 1. Generate meaningful synthetic data
num_users = 200
num_features = 20
num_classes = 5

# Career feature templates
class_templates = torch.zeros(num_classes, num_features)
for i in range(num_classes):
    # Each career class is associated with 3 features
    class_templates[i, i*3 : (i+1)*3] = 1  

# Base noise for features
x = torch.randn(num_users, num_features) * 0.2
# Career labels
y = torch.randint(0, num_classes, (num_users,))
# Inject career-related features
for i in range(num_users):
    x[i] += class_templates[y[i]]

# Generate edges and convert to undirected graph
edge_index = torch.randint(0, num_users, (2, 300))
edge_index = torch.unique(edge_index, dim=1)  # remove duplicates
edge_index = to_undirected(edge_index)        # make graph undirected

# Split dataset into training and test sets
train_mask = torch.zeros(num_users, dtype=torch.bool)
train_mask[:160] = True
test_mask = torch.zeros(num_users, dtype=torch.bool)
test_mask[160:] = True

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    test_mask=test_mask
)

class CareerMPNN(MessagePassing):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__(aggr='mean')
        # 1) Input projection
        self.lin_x = nn.Linear(num_features, hidden_dim)
        # Message and update networks both use hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        # First map x to hidden_dim
        h = F.relu(self.lin_x(x))
        # Two rounds of message passing
        for _ in range(2):
            h = self.propagate(edge_index, h=h)
        return self.classifier(h)

    def message(self, h_i, h_j):
        # h_i and h_j are both [*, hidden_dim]
        return self.message_mlp(torch.cat([h_i, h_j], dim=-1))

    def update(self, aggr_out):
        return self.update_mlp(aggr_out)

# 3. Training and evaluation
model = CareerMPNN(num_features=20, hidden_dim=10, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = accuracy_score(
            data.y[data.test_mask].numpy(),
            pred[data.test_mask].numpy()
        )
    return acc

# Training loop
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        acc = test()
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

import matplotlib.pyplot as plt

# Inference to obtain predicted labels
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1).cpu().numpy()

# Test set indices and data
test_idx = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
x_test = data.x[test_idx].cpu().numpy()    # shape (40, num_features)
y_true = data.y[test_idx].cpu().numpy()    # shape (40,)

# Plot feature distributions for each test sample
for i, sample_idx in enumerate(test_idx):
    plt.figure(figsize=(4, 2))
    plt.plot(x_test[i])
    plt.xlabel('Feature Index (0–19)')
    plt.ylabel('Feature Value')
    plt.title(f'Sample {sample_idx}: True {y_true[i]+1}, Pred {preds[sample_idx]+1}')
    plt.tight_layout()
    plt.show()

