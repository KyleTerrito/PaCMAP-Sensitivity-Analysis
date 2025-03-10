import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import datetime

# Create results directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f"Results_SOM/SOM_Advanced_{timestamp}"
os.makedirs(results_folder, exist_ok=True)
print(f"Created directory: {results_folder}")

# Load original data
file_path = os.path.join("DATA", "molecular_descriptors_concat_data_product2_Target-Logcmc.csv")
# Alternative: use processed data if original not available
# file_path = "original_data.csv"
df = pd.read_csv(file_path)

# Store names and targets
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

# Define SOM parameters - using 24x28 grid to match your examples
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
win_map = som.win_map(data_scaled)
coordinate_map = np.array([som.winner(x) for x in data_scaled])

# Get distances from each node to its neighbors (U-matrix)
distance_map = som.distance_map()

# Perform k-means clustering on the data points to get groups
# We'll use 4 clusters to match your example images
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(data_scaled)

# Save cluster assignments
cluster_results = pd.DataFrame({
    'Name': names,
    'TARGET': targets,
    'SOM_X': coordinate_map[:, 0],
    'SOM_Y': coordinate_map[:, 1],
    'Cluster': cluster_labels
})
cluster_results.to_csv(os.path.join(results_folder, "som_clusters.csv"), index=False)

# Create the base U-Matrix plot function
def plot_umatrix_with_points(points_to_show=None, title="SOM U-Matrix", filename="SOM_UMatrix.png"):
    """
    Plot the U-Matrix with optional highlighted points
    
    Args:
        points_to_show: List of indices of points to highlight, or None for no points
        title: Plot title
        filename: Filename to save the plot
    """
    plt.figure(figsize=(10, 12))
    plt.pcolor(distance_map.T, cmap='Blues_r')  # Use Blues_r to match your examples
    plt.colorbar(label='Distance')
    
    # If we have points to highlight, add them to the plot
    if points_to_show is not None:
        for idx in points_to_show:
            x, y = coordinate_map[idx]
            plt.plot(x + 0.5, y + 0.5, 'wo', markersize=8, markeredgecolor='w')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename))
    plt.close()

# Plot the base U-Matrix with no points (like your Image 1)
plot_umatrix_with_points(points_to_show=None, 
                        title="SOM U-Matrix", 
                        filename="SOM_UMatrix_Base.png")

# Plot U-Matrix with points from each cluster separately (like Images 2-4)
for cluster_id in range(4):
    # Get indices of points in this cluster
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    
    plot_umatrix_with_points(
        points_to_show=cluster_indices,
        title=f"SOM U-Matrix - Cluster {cluster_id} Points",
        filename=f"SOM_UMatrix_Cluster{cluster_id}.png"
    )

# Create visualization with all points colored by cluster
plt.figure(figsize=(10, 12))
plt.pcolor(distance_map.T, cmap='Blues_r')
plt.colorbar(label='Distance')

# Add points colored by cluster
cluster_colors = ['white', 'cyan', 'yellow', 'magenta']  # Distinct colors for clusters
for cluster_id in range(4):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    for idx in cluster_indices:
        x, y = coordinate_map[idx]
        plt.plot(x + 0.5, y + 0.5, 'o', color=cluster_colors[cluster_id], 
                markersize=8, markeredgecolor='black')

plt.title("SOM U-Matrix with All Clusters")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_UMatrix_AllClusters.png"))
plt.close()

# Create a version with Square markers instead of circles
plt.figure(figsize=(10, 12))
plt.pcolor(distance_map.T, cmap='Blues_r')
plt.colorbar(label='Distance')

# Add points with square markers
marker_types = ['o', 's', 'D', '^']  # Circle, square, diamond, triangle
for cluster_id in range(4):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    for idx in cluster_indices:
        x, y = coordinate_map[idx]
        plt.plot(x + 0.5, y + 0.5, marker_types[cluster_id], color='white', 
                markersize=8, markeredgecolor='white')

plt.title("SOM U-Matrix with Different Marker Types")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "SOM_UMatrix_DifferentMarkers.png"))
plt.close()

# Analyze TARGET values by cluster
cluster_target_stats = cluster_results.groupby('Cluster')['TARGET'].agg(['mean', 'min', 'max', 'count'])
print("TARGET statistics by cluster:")
print(cluster_target_stats)
cluster_target_stats.to_csv(os.path.join(results_folder, "cluster_target_stats.csv"))

# Try another approach: Divide points into quadrants based on TARGET values
q1, q2, q3 = np.percentile(targets, [25, 50, 75])
print(f"TARGET quartiles: Q1={q1}, Q2={q2}, Q3={q3}")

# Assign quartile labels 
quartile_labels = pd.cut(targets, 
                        bins=[targets.min()-0.1, q1, q2, q3, targets.max()+0.1], 
                        labels=[0, 1, 2, 3])

# Create quartile visualization
plt.figure(figsize=(10, 12))
plt.pcolor(distance_map.T, cmap='Blues_r')
plt.colorbar(label='Distance')

# Plot points from each quartile
for quartile in range(4):
    quartile_indices = np.where(quartile_labels == quartile)[0]
    for idx in quartile_indices:
        x, y = coordinate_map[idx]
        plt.plot(x + 0.5, y + 0.5, 'wo', markersize=8, markeredgecolor='w')
    
    # Save individual quartile plots
    plot_umatrix_with_points(
        points_to_show=quartile_indices,
        title=f"SOM U-Matrix - TARGET Quartile {quartile+1}",
        filename=f"SOM_UMatrix_Quartile{quartile+1}.png"
    )

# Save quartile information
quartile_results = pd.DataFrame({
    'Name': names,
    'TARGET': targets,
    'SOM_X': coordinate_map[:, 0],
    'SOM_Y': coordinate_map[:, 1],
    'TARGET_Quartile': quartile_labels
})
quartile_results.to_csv(os.path.join(results_folder, "som_target_quartiles.csv"), index=False)

# You can also try to manually define regions on the SOM grid
# This gives you the most control to match specific patterns

# Define several regions on the SOM (customize these coordinates based on your U-matrix patterns)
regions = {
    'Region1': [(x, y) for x in range(5, 15) for y in range(5, 10)],
    'Region2': [(x, y) for x in range(15, 20) for y in range(15, 20)],
    'Region3': [(x, y) for x in range(0, 5) for y in range(20, 25)],
    'Region4': [(x, y) for x in range(20, 23) for y in range(0, 5)]
}

# Find points in each region
for region_name, region_coords in regions.items():
    region_indices = []
    for i, (x, y) in enumerate(coordinate_map):
        if (x, y) in region_coords:
            region_indices.append(i)
    
    if region_indices:
        plot_umatrix_with_points(
            points_to_show=region_indices,
            title=f"SOM U-Matrix - {region_name}",
            filename=f"SOM_UMatrix_{region_name}.png"
        )

print(f"All SOM visualizations saved to {results_folder}")