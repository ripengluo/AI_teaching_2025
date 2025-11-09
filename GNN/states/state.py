import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

def generate_state_data(state_id, is_blue):
    """Generate data for one state"""
    np.random.seed(state_id)
    
    # Create graph with 8 cities
    G = nx.Graph()
    num_cities = 8
    
    # Base feature generation with state bias
    if is_blue:
        education_bias = 0.6
        agri_bias = -0.3
        economy_bias = 0.4
        diversity_bias = 0.5
        aging_bias = -0.4
    else:
        education_bias = 0.3
        agri_bias = 0.4
        economy_bias = 0.1
        diversity_bias = -0.3
        aging_bias = 0.3
    
    # Generate node features with some randomness
    for i in range(num_cities):
        education = np.clip(np.random.normal(0.5 + education_bias, 0.15), 0, 1)
        agriculture = np.clip(np.random.normal(0.5 + agri_bias, 0.2), 0, 1)
        economy = np.clip(np.random.normal(0.5 + economy_bias, 0.2), 0, 1)
        diversity = np.clip(np.random.normal(0.5 + diversity_bias, 0.15), 0, 1)
        aging = np.clip(np.random.normal(0.5 + aging_bias, 0.15), 0, 1)
        
        G.add_node(i, features=np.array([education, agriculture, economy, diversity, aging]))
    
    # Create edges (transportation links)
    # Blue states tend to have more connections
    num_edges = np.random.randint(8, 15) if is_blue else np.random.randint(5, 12)
    
    possible_edges = [(i, j) for i in range(num_cities) for j in range(i+1, num_cities)]
    selected_edges = np.random.choice(len(possible_edges), num_edges, replace=False)
    
    for edge_idx in selected_edges:
        i, j = possible_edges[edge_idx]
        # Transportation volume influenced by economy and distance
        distance = np.random.uniform(0.1, 1.0)
        economy_factor = (G.nodes[i]['features'][2] + G.nodes[j]['features'][2]) / 2
        transport = np.clip(np.random.normal(0.5 + 0.3*economy_factor - 0.2*distance, 0.1), 0, 1)
        G.add_edge(i, j, features=np.array([transport, distance]))
    
    return G

def generate_dataset(num_states=20):
    """Generate dataset of multiple states"""
    states = []
    labels = []
    
    # Generate half blue, half red states
    for i in range(num_states):
        is_blue = i % 2 == 0  # Alternate for balanced dataset
        state_graph = generate_state_data(i, is_blue)
        states.append(state_graph)
        labels.append(1 if is_blue else 0)
    
    return states, np.array(labels)

# Generate the dataset
states, labels = generate_dataset(20)

print(labels)
