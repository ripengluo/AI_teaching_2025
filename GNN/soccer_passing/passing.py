import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse

# Node and edge feature dimensions
node_feature_dim = 5
edge_feature_dim = 2
num_players = 11

# Mapping from node index to soccer position
positions = [
    "Goalkeeper",              # 0
    "Center Defender",         # 1
    "Left Defender",           # 2
    "Right Defender",          # 3
    "Sweeper",                 # 4
    "Defensive Midfielder",    # 5
    "Center Midfielder",       # 6
    "Attacking Midfielder",    # 7
    "Left Winger",             # 8
    "Striker",                 # 9
    "Right Winger"             # 10
]

# Function to generate graph data
def generate_soccer_graph():
    base_features = torch.tensor([
        [1, 0.9, 0.4, 0.9, 0.9],
        [1, 0.8, 0.6, 0.85, 0.9],
        [1, 0.8, 0.65, 0.88, 0.88],
        [1, 0.85, 0.75, 0.83, 0.85],
        [1, 0.85, 0.75, 0.83, 0.85],
        [2, 0.95, 0.8, 0.75, 0.8],
        [2, 0.95, 0.85, 0.7, 0.8],
        [2, 0.95, 0.85, 0.7, 0.8],
        [3, 0.7, 0.95, 0.65, 0.85],
        [3, 0.7, 0.9, 0.7, 0.85],
        [3, 0.7, 0.9, 0.68, 0.85]
    ], dtype=torch.float)

    noise = 0.2 * torch.randn_like(base_features)
    node_features = base_features + noise

    adjacency = torch.ones((num_players, num_players)) - torch.eye(num_players)
    edge_index, _ = dense_to_sparse(adjacency)

    edge_features = []
    edge_labels = []
    for i, j in zip(edge_index[0], edge_index[1]):
        # Assign distance based on team structure
        if (i == 0 and 1 <= j <= 4) or (1 <= i <= 4 and 1 <= j <= 4):
            distance = 0.2
        elif (1 <= i <= 4 and j == 5) or (i == 5 and 6 <= j <= 7) or (6 <= i <= 7 and 8 <= j <= 10):
            distance = 0.5
        else:
            distance = 0.8
        chemistry = 0.9 if (i % 2 == j % 2) else 0.3
        edge_features.append([distance, chemistry])

        defense_pressure = node_features[j][0].item()
        stamina = node_features[j][4].item()

        # Construct nonlinear raw score, ensuring distribution falls within [0.2, 0.8]
        raw_score = (
            -10.0 * defense_pressure -
            10.0 * distance +
            15.0 * (chemistry ** 2) +
            20.0 * (stamina ** 2) +
            5.0 * torch.randn(1).item()
        )
        pass_prob = torch.sigmoid(torch.tensor(raw_score))
        scaled_prob = 0.2 + 0.6 * pass_prob.item()
        edge_labels.append(scaled_prob)

    edge_features = torch.tensor(edge_features, dtype=torch.float)
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=edge_labels)

# Prepare training data loader
tain_dataset = [generate_soccer_graph() for _ in range(100)]
loader = DataLoader(tain_dataset, batch_size=10)

class MPNNEdgePredictor(MessagePassing):
    def __init__(self, node_in, edge_in, hidden_dim=32):
        super(MPNNEdgePredictor, self).__init__(aggr='mean')  # Neighbor aggregation method
        
        # Node and edge feature encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.ReLU()
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.ReLU()
        )
        
        # Message function: MLP(h_i || h_j || e_ij)
        self.message_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Update function: Just use a MLP layer
        self.update_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        
        # Edge prediction head: MLP(h_i || h_j || e_ij)
        self.edge_pred = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        # Encode initial features
        h = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        # Multiple rounds of message passing (e.g., T=2)
        for _ in range(2):
            h = self.propagate(edge_index, h=h, edge_attr=edge_attr_encoded)
        
        # Edge prediction (using final node embeddings)
        row, col = edge_index
        edge_feats = torch.cat([h[row], h[col], edge_attr_encoded], dim=-1)
        return self.edge_pred(edge_feats).squeeze(-1)

    def message(self, h_i, h_j, edge_attr):
        # h_i: target node, h_j: source node
        return self.message_mlp(torch.cat([h_i, h_j, edge_attr], dim=-1))

    def update(self, aggr_out, h):
        # Update node states with GRU
        return self.update_mlp(aggr_out)

# Initialize model and optimizer
model = MPNNEdgePredictor(node_feature_dim, edge_feature_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(100):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")

# Evaluation on a new test graph
test_graph = generate_soccer_graph()
model.eval()
with torch.no_grad():
    predictions = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)

central_defender_index = 1
# Find edges originating from the Central Defender node
cd_edges = (test_graph.edge_index[0] == central_defender_index).nonzero(as_tuple=False).view(-1)
pass_probs = predictions[cd_edges]

# Print pass probabilities with position names
for target_idx, prob in zip(test_graph.edge_index[1, cd_edges], pass_probs):
    print(f'Pass from {positions[central_defender_index]} to {positions[target_idx.item()]}: Probability {prob.item():.4f}')

