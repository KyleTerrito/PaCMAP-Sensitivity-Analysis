import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import os
import datetime

# Create results directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f"Results_SOM/SOM_{timestamp}"
os.makedirs(results_folder, exist_ok=True)
print(f"Created directory: {results_folder}")

# Load original data
file_path = os.path.join("DATA", "molecular_descriptors_concat_data_product2_Target-Logcmc.csv")
# If you don't have the original file, use the processed data:
# file_path = "original_data.csv"
df = pd.read_csv(file_path)

# Store names and targets before dropping them
names = df['Name']
targets = df['TARGET']

# Drop non-feature columns
columns_to_drop = ['Name', 'TARGET']
if 'Index' in df.columns:
    columns_to_drop.append('Index')
if 'SMILES' in df.columns:
    columns_to_drop.append('SMILES')
if 'smiles' in df.columns:
    columns_to_drop.append('smiles')
if 'Smiles' in df.columns:
    columns_to_drop.append('Smiles')
if 'cmc' in df.columns:
    columns_to_drop.append('cmc')
elif 'Log(cmc*1000+1)' in df.columns:
    columns_to_drop.append('Log(cmc*1000+1)')

# Make a copy of the data before dropping columns
df_features = df.copy()
df_features.drop(columns=columns_to_drop, axis=1, inplace=True)

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_features)

# Define SOM parameters
som_x_dim = 10  # width of the SOM grid
som_y_dim = 10  # height of the SOM grid
input_len = data_scaled.shape[1]  # number of features

# Initialize and train the SOM
# sigma is the initial neighborhood radius
# learning_rate is how fast the SOM adapts
som = MiniSom(x=som_x_dim, y=som_y_dim, input_len=input_len, 
              sigma=1.0, learning_rate=0.5, 
              neighborhood_function='gaussian', random_seed=42)

# Initialize weights
som.random_weights_init(data_scaled)
print("Training SOM...")

# Train the SOM
# num_iteration is how many times the SOM will see all the data
som.train(data_scaled, num_iteration=5000, verbose=True)
print("SOM training completed")

# Get the coordinates of each data point on the SOM
win_map = som.win_map(data_scaled)
coordinate_map = np.array([som.winner(x) for x in data_scaled])

# Get distances from each node to its neighbors
distance_map = som.distance_map()

# Plot the U-Matrix (Unified Distance Matrix)
plt.figure(figsize=(10, 8))
plt.pcolor(distance_map.T, cmap='bone_r')  # Transposed for correct orientation
plt.colorbar(label='Distance')
plt.title('SOM U-Matrix')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_UMatrix.png"))

# Plot SOM with data points colored by TARGET values
plt.figure(figsize=(12, 10))
plt.pcolor(distance_map.T, cmap='bone_r', alpha=0.5)
plt.colorbar(label='Distance')

# Get unique coordinates to avoid overlapping points
unique_coordinates = {}
for i, (x, y) in enumerate(coordinate_map):
    if (x, y) not in unique_coordinates:
        unique_coordinates[(x, y)] = []
    unique_coordinates[(x, y)].append(i)

# Plot points on SOM grid
for (x, y), indices in unique_coordinates.items():
    # Calculate average TARGET value for points mapped to this node
    avg_target = np.mean([targets.iloc[i] for i in indices])
    plt.scatter(x+0.5, y+0.5, c=avg_target, cmap='viridis', 
                s=100, marker='o', edgecolors='black')

plt.title('SOM with Data Points (Colored by TARGET)')
plt.xlabel('SOM X')
plt.ylabel('SOM Y')
plt.colorbar(label='TARGET Values')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_ColoredByTarget.png"))

# Add data points to plot with jitter to see overlapping points
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(coordinate_map):
    # Add small jitter to show overlapping points
    jitter_x = np.random.normal(0, 0.1)
    jitter_y = np.random.normal(0, 0.1)
    plt.scatter(x+0.5+jitter_x, y+0.5+jitter_y, c=targets.iloc[i], 
                cmap='viridis', s=50, marker='o', edgecolors='black')
    
plt.colorbar(label='TARGET Values')
plt.title('SOM with Individual Data Points (with jitter)')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_IndividualPoints.png"))

# Create a DataFrame with SOM coordinates for each molecule
som_results = pd.DataFrame({
    'Name': names,
    'TARGET': targets,
    'SOM_X': coordinate_map[:, 0],
    'SOM_Y': coordinate_map[:, 1]
})

# Save the SOM coordinates to CSV
som_results.to_csv(os.path.join(results_folder, "som_mapping.csv"), index=False)

# Create a feature heatmap for each weight component
plt.figure(figsize=(15, 12))
feature_names = df_features.columns.tolist()
nrows = int(np.ceil(len(feature_names) / 4))  # 4 features per row

for i, feature in enumerate(feature_names):
    plt.subplot(nrows, 4, i+1)
    weights = som.get_weights()[:, :, i]
    plt.pcolor(weights.T, cmap='coolwarm')
    plt.title(f'Feature: {feature}')
    plt.colorbar()
    
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_FeatureHeatmaps.png"))

# Calculate and plot component planes
plt.figure(figsize=(20, 15))
feature_names = df_features.columns.tolist()
num_features = len(feature_names)
nrows = int(np.ceil(num_features / 4))

for i, feature in enumerate(feature_names):
    plt.subplot(nrows, 4, i+1)
    weights = som.get_weights()[:, :, i]
    plt.pcolor(weights.T, cmap='coolwarm')
    plt.title(f'Component Plane: {feature}')
    plt.colorbar()
    
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_ComponentPlanes.png"))

# Perform K-means clustering on the SOM grid
from sklearn.cluster import KMeans

# Get weights from the trained SOM model
weights = som.get_weights()
n_clusters = 5  # You can adjust this based on your data

# Reshape the weights to perform K-means
reshaped_weights = weights.reshape(-1, input_len)

# Perform K-means clustering on the SOM weights
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(reshaped_weights)

# Reshape cluster labels to match SOM grid
cluster_map = cluster_labels.reshape(som_x_dim, som_y_dim)

# Plot SOM with clusters
plt.figure(figsize=(12, 10))
plt.pcolor(cluster_map.T, cmap='tab10', alpha=0.75)
plt.colorbar(label='Cluster')
plt.title('SOM Grid Clustered with K-means')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_Clusters.png"))

# Assign cluster labels to each data point based on its SOM position
point_clusters = np.array([cluster_map[x, y] for x, y in coordinate_map])

# Create a DataFrame with cluster assignments
cluster_results = pd.DataFrame({
    'Name': names,
    'TARGET': targets,
    'SOM_X': coordinate_map[:, 0],
    'SOM_Y': coordinate_map[:, 1],
    'Cluster': point_clusters
})

# Save the cluster assignments to CSV
cluster_results.to_csv(os.path.join(results_folder, "som_clusters.csv"), index=False)

print(f"All SOM analysis results saved to {results_folder}")

# Visualize both the DBSCAN clusters and SOM clusters for comparison
# Load DBSCAN cluster labels if available
try:
    # Try to load from the same directory as the script or adjust path as needed
    dbscan_labels = pd.read_csv("cluster_labels.csv")["Cluster Labels"]
    reduced_data = pd.read_csv("reduced_data.csv")
    
    # Plot comparison of clustering methods
    plt.figure(figsize=(15, 5))
    
    # DBSCAN clusters on PaCMAP
    plt.subplot(1, 3, 1)
    plt.scatter(reduced_data["Dim1"], reduced_data["Dim2"], 
                c=dbscan_labels, cmap='tab10', s=50, alpha=0.75)
    plt.title('DBSCAN Clusters on PaCMAP')
    plt.xlabel('PaCMAP Component 1')
    plt.ylabel('PaCMAP Component 2')
    plt.colorbar(label='DBSCAN Cluster')
    
    # SOM-based clusters on PaCMAP
    plt.subplot(1, 3, 2)
    plt.scatter(reduced_data["Dim1"], reduced_data["Dim2"], 
                c=point_clusters, cmap='tab10', s=50, alpha=0.75)
    plt.title('SOM-based Clusters on PaCMAP')
    plt.xlabel('PaCMAP Component 1')
    plt.ylabel('PaCMAP Component 2')
    plt.colorbar(label='SOM Cluster')
    
    # SOM grid with clusters
    plt.subplot(1, 3, 3)
    plt.pcolor(cluster_map.T, cmap='tab10', alpha=0.75)
    plt.title('SOM Grid Clusters')
    plt.colorbar(label='Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "Clustering_Comparison.png"))
    
except Exception as e:
    print(f"Could not load DBSCAN clusters for comparison: {e}")