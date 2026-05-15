import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Generate music data for 3 styles (100 songs each)
np.random.seed(42)
styles = {
    "Rock":     {'bpm': (120, 160), 'intensity': (0.7, 1.0), 'melody': (0.3, 0.6)},
    "Jazz":     {'bpm': (60, 100),  'intensity': (0.4, 0.7), 'melody': (0.7, 1.0)},
    "Electronic": {'bpm': (100, 140), 'intensity': (0.6, 0.9), 'melody': (0.1, 0.5)}
}

data = []
labels = []
for style, params in styles.items():
    bpm = np.random.uniform(params['bpm'][0], params['bpm'][1], 100)
    intensity = np.random.uniform(params['intensity'][0], params['intensity'][1], 100)
    melody = np.random.uniform(params['melody'][0], params['melody'][1], 100)
    data.append(np.column_stack([bpm, intensity, melody]))
    labels.extend([style] * 100)

X = np.vstack(data)  # Combine all music data
y = np.array(labels)  # True labels (only for validation, unsupervised learning doesn't use this)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled.shape)

# Visualize original data
fig = plt.figure(figsize=(12, 4))
for i, style in enumerate(styles.keys()):
    plt.scatter(X_scaled[y==style, 0], X_scaled[y==style, 1], label=style, alpha=0.6)
plt.xlabel("BPM (Standardized)")
plt.ylabel("Intensity (Standardized)")
plt.title("Original Music Data Distribution")
plt.legend()
plt.show()


# Define a simple MLP as Encoder
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),  # Input layer → Hidden layer
            nn.ReLU(),
            nn.Linear(16, latent_dim)  # Hidden layer → Latent space (2D for visualization)
        )

    def forward(self, x):
        return self.layers(x)

encoder = MLPEncoder()
#print(encoder)

# Autoencoder: Encoder + Decoder
class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Output dimension = input dimension
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Training
model = Autoencoder(encoder)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.FloatTensor(X_scaled)
epochs = 500
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, X_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot training loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("Autoencoder Training Loss")
plt.show()

with torch.no_grad():
    latent_features = encoder(X_tensor).numpy()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert to numerical labels (0, 1, 2...)

# Visualize latent space
plt.scatter(latent_features[:, 0], latent_features[:, 1], c=y_encoded, alpha=0.6)
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent Space Extracted by MLP Encoder")
plt.colorbar(ticks=range(3), label='True Style')
plt.show()


# Clustering (assuming we know there are 3 styles)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(latent_features)

# Visualize clustering results
plt.scatter(latent_features[:, 0], latent_features[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Cluster Centers')
plt.title("Clustering Results in Latent Space")
plt.legend()
plt.show()

# Count the most frequent true style in each cluster
cluster_to_style = {}
for cluster_id in range(3):
    true_labels_in_cluster = y[clusters == cluster_id]
    style = max(set(true_labels_in_cluster), key=list(true_labels_in_cluster).count)
    cluster_to_style[cluster_id] = style
    print(f"Cluster {cluster_id} → {style}")
# Example output:
# Cluster 0 → Rock
# Cluster 1 → Jazz
# Cluster 2 → Electronic

def recommend_song(bpm, intensity, melody):
    # Standardize input
    input_scaled = scaler.transform([[bpm, intensity, melody]])
    
    # Extract latent features and predict cluster
    with torch.no_grad():
        latent = encoder(torch.FloatTensor(input_scaled)).numpy()
    cluster = kmeans.predict(latent)[0]
    
    # Return recommended style and similar songs
    recommended_style = cluster_to_style[cluster]
    print(f"Recommended style: {recommended_style}")
    print("Similar song examples:")
    similar_songs = X_scaled[clusters == cluster][:3]  # Take first 3 songs from this cluster
    return scaler.inverse_transform(similar_songs)  # Revert to original BPM/Intensity/Melody

# Example: User likes fast-paced, high-intensity music
print(recommend_song(bpm=100, intensity=0.7, melody=0.7))
