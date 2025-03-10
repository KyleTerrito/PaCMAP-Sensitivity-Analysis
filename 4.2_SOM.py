import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import os
import datetime

# Your specific results folder path
input_folder = r"RESULTS\2025-03-09_20-03-29-logcmc"


# Create results directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f"Results_SOM/SOM_PaCMAP_{timestamp}"
os.makedirs(results_folder, exist_ok=True)
print(f"Created directory: {results_folder}")

# Load your existing PaCMAP results and cluster data from the specific folder
names = pd.read_csv(os.path.join(input_folder, "names.csv"))["Name"]
targets = pd.read_csv(os.path.join(input_folder, "targets.csv"))["TARGET"]
reduced_data = pd.read_csv(os.path.join(input_folder, "reduced_data.csv"))
original_data = pd.read_csv(os.path.join(input_folder, "original_data.csv"))
pacmap_clusters = pd.read_csv(os.path.join(input_folder, "cluster_labels.csv"))["Cluster Labels"]

# Extract features for SOM (drop non-feature columns from original data)
columns_to_drop = ['Name', 'TARGET', 'Index', 'cmc']
columns_to_drop = [col for col in columns_to_drop if col in original_data.columns]
df_features = original_data.copy()
df_features.drop(columns=columns_to_drop, axis=1, inplace=True)

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_features)

# Define SOM parameters
som_x_dim = 24
som_y_dim = 28
input_len = data_scaled.shape[1]

# Initialize and train the SOM
som = MiniSom(x=som_x_dim, y=som_y_dim, input_len=input_len, 
              sigma=2.0, learning_rate=0.5, 
              neighborhood_function='gaussian', random_seed=42)

# Initialize weights
som.random_weights_init(data_scaled)
print("Training SOM...")

# Train the SOM
som.train(data_scaled, num_iteration=10000, verbose=True)
print("SOM training completed")

# Get the coordinates of each data point on the SOM
coordinate_map = np.array([som.winner(x) for x in data_scaled])

# Get distances from each node to its neighbors (U-matrix)
distance_map = som.distance_map()

# Plot the base U-Matrix
plt.figure(figsize=(10, 12))
plt.pcolor(distance_map.T, cmap='Blues_r')
plt.colorbar(label='Distance')
plt.title("SOM U-Matrix (Base)")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_UMatrix_Base.png"))
plt.close()

# Create a DataFrame to store all results
all_results = pd.DataFrame({
    'Name': names,
    'TARGET': targets,
    'SOM_X': coordinate_map[:, 0],
    'SOM_Y': coordinate_map[:, 1],
    'PaCMAP_Dim1': reduced_data['Dim1'],
    'PaCMAP_Dim2': reduced_data['Dim2'],
    'PaCMAP_Cluster': pacmap_clusters
})

# Save the combined results
all_results.to_csv(os.path.join(results_folder, "combined_results.csv"), index=False)

# Create U-Matrix plots highlighting each PaCMAP cluster separately
# This will create similar visualizations to your examples
unique_clusters = np.unique(pacmap_clusters)

for cluster_id in unique_clusters:
    # Get indices for this cluster
    cluster_indices = np.where(pacmap_clusters == cluster_id)[0]
    
    # Plot U-Matrix with this cluster's points highlighted
    plt.figure(figsize=(10, 12))
    plt.pcolor(distance_map.T, cmap='coolwarm')
    plt.colorbar(label='Distance')
    
    for idx in cluster_indices:
        x, y = coordinate_map[idx]
        plt.plot(x + 0.5, y + 0.5, 'wo', markersize=8, markeredgecolor='w')
    
    plt.title(f"SOM U-Matrix - PaCMAP Cluster {cluster_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"SOM_UMatrix_PaCMAP_Cluster{cluster_id}.png"))
    plt.close()

# Create TARGET quartile visualizations (splitting TARGET into 4 groups)
q1, q2, q3 = np.percentile(targets, [25, 50, 75])
quartile_indices = {
    0: np.where(targets <= q1)[0],
    1: np.where((targets > q1) & (targets <= q2))[0],
    2: np.where((targets > q2) & (targets <= q3))[0],
    3: np.where(targets > q3)[0]
}

# Plot each TARGET quartile separately
for quartile, indices in quartile_indices.items():
    plt.figure(figsize=(10, 12))
    plt.pcolor(distance_map.T, cmap='Blues_r')
    plt.colorbar(label='Distance')
    
    for idx in indices:
        x, y = coordinate_map[idx]
        plt.plot(x + 0.5, y + 0.5, 'wo', markersize=8, markeredgecolor='w')
    
    plt.title(f"SOM U-Matrix - TARGET Quartile {quartile+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"SOM_UMatrix_TARGET_Quartile{quartile+1}.png"))
    plt.close()

# Create a visualization showing all PaCMAP clusters on the U-Matrix with different markers
plt.figure(figsize=(10, 12))
plt.pcolor(distance_map.T, cmap='Blues_r')
plt.colorbar(label='Distance')

markers = ['o', 's', 'D', '^', 'v']  # Different marker shapes for each cluster
for cluster_id in unique_clusters:
    cluster_indices = np.where(pacmap_clusters == cluster_id)[0]
    marker = markers[cluster_id % len(markers)]
    
    for idx in cluster_indices:
        x, y = coordinate_map[idx]
        plt.plot(x + 0.5, y + 0.5, marker, color='white', 
                markersize=8, markeredgecolor='white')

plt.title("SOM U-Matrix with All PaCMAP Clusters")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_UMatrix_All_PaCMAP_Clusters.png"))
plt.close()

# Create component planes to show how features are distributed
features = df_features.columns.tolist()
num_features = len(features)
nrows = int(np.ceil(num_features / 4))  # 4 features per row

plt.figure(figsize=(15, nrows * 3))
for i, feature in enumerate(features):
    plt.subplot(nrows, 4, i+1)
    feature_weights = som.get_weights()[:, :, i]
    plt.pcolor(feature_weights.T, cmap='coolwarm')
    plt.title(f'Feature: {feature}')
    plt.colorbar()
    
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_ComponentPlanes.png"))
plt.close()

# Create visualizations showing both PaCMAP and SOM
# This creates a side-by-side view of PaCMAP clusters and how they map to the SOM
plt.figure(figsize=(18, 8))

# Left: PaCMAP plot with clusters
plt.subplot(1, 2, 1)
for cluster_id in unique_clusters:
    mask = pacmap_clusters == cluster_id
    plt.scatter(reduced_data.loc[mask, 'Dim1'], reduced_data.loc[mask, 'Dim2'], 
                label=f'Cluster {cluster_id}', alpha=0.75)

plt.xlabel("PaCMAP Component 1")
plt.ylabel("PaCMAP Component 2")
plt.title("PaCMAP Clustering")
plt.legend()
plt.grid(True)

# Right: SOM U-Matrix with same cluster colors
plt.subplot(1, 2, 2)
plt.pcolor(distance_map.T, cmap='coolwarm', alpha=0.5)

# Add colored points for each cluster
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
for i, cluster_id in enumerate(unique_clusters):
    cluster_indices = np.where(pacmap_clusters == cluster_id)[0]
    for idx in cluster_indices:
        x, y = coordinate_map[idx]
        plt.plot(x + 0.5, y + 0.5, 'o', color=colors[i], 
                markersize=8, markeredgecolor='black')

plt.title("SOM U-Matrix with PaCMAP Clusters")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "PaCMAP_vs_SOM_Clusters.png"))
plt.close()

# Plot PaCMAP clusters on the SOM grid as a heatmap
for cluster_id in unique_clusters:
    # Create a grid to show where points from this cluster appear on the SOM
    cluster_grid = np.zeros((som_x_dim, som_y_dim))
    cluster_indices = np.where(pacmap_clusters == cluster_id)[0]
    
    for idx in cluster_indices:
        x, y = coordinate_map[idx]
        if x < som_x_dim and y < som_y_dim:
            cluster_grid[x, y] += 1
    
    # Plot the density of this cluster on the SOM
    plt.figure(figsize=(10, 12))
    plt.pcolor(cluster_grid.T, cmap='Reds')
    plt.colorbar(label='Number of Points')
    plt.title(f"SOM Grid Density - PaCMAP Cluster {cluster_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"SOM_Density_PaCMAP_Cluster{cluster_id}.png"))
    plt.close()

# Create image similar to your examples with white points in specific regions
for i in range(4):  # Create 4 different images with points in different regions
    plt.figure(figsize=(10, 12))
    plt.pcolor(distance_map.T, cmap='Blues_r')
    plt.colorbar(label='Distance')
    
    # Choose a different set of points for each image
    # Image 1: Top 25% of data points by PaCMAP Dim1
    if i == 0:
        sorted_indices = np.argsort(reduced_data['Dim1'])
        show_indices = sorted_indices[-len(sorted_indices)//4:]
        title = "SOM U-Matrix - Top 25% by PaCMAP Dim1"
    # Image 2: Bottom 25% of data points by PaCMAP Dim1
    elif i == 1:
        sorted_indices = np.argsort(reduced_data['Dim1'])
        show_indices = sorted_indices[:len(sorted_indices)//4]
        title = "SOM U-Matrix - Bottom 25% by PaCMAP Dim1"
    # Image 3: Top 25% of data points by PaCMAP Dim2
    elif i == 2:
        sorted_indices = np.argsort(reduced_data['Dim2'])
        show_indices = sorted_indices[-len(sorted_indices)//4:]
        title = "SOM U-Matrix - Top 25% by PaCMAP Dim2"
    # Image 4: Bottom 25% of data points by PaCMAP Dim2
    else:
        sorted_indices = np.argsort(reduced_data['Dim2'])
        show_indices = sorted_indices[:len(sorted_indices)//4]
        title = "SOM U-Matrix - Bottom 25% by PaCMAP Dim2"
    
    # Plot the selected points
    for idx in show_indices:
        x, y = coordinate_map[idx]
        plt.plot(x + 0.5, y + 0.5, 'wo', markersize=8, markeredgecolor='w')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"SOM_UMatrix_Example{i+1}.png"))
    plt.close()

print(f"All SOM visualizations saved to {results_folder}")