import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pacmap
from sklearn.cluster import DBSCAN
import datetime
from sklearn.preprocessing import StandardScaler

# Create a timestamped results folder
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f"SENSITIVITY/{timestamp}"
os.makedirs(results_folder, exist_ok=True)
print(f"Created directory: {results_folder}")

# Load data
# file_path = os.path.join("DATA", "molecular_descriptors.csv")
# file_path = os.path.join("DATA", "molecular_descriptors_concat_data_product.csv")
# file_path = os.path.join("DATA", "molecular_descriptors_concat_data_product2.csv")
file_path = os.path.join("DATA", "molecular_descriptors_concat_data_product2_Target-Logcmc.csv")
# file_path = os.path.join("DATA", "molecular_descriptors_concat_data_produc2_Target-cmc.csv")
df = pd.read_csv(file_path)

#drop non-feature columns
if 'Name' in df.columns:
    df.drop(['Name'], axis=1, inplace=True)
if 'TARGET' in df.columns:
    df.drop(['TARGET'], axis=1, inplace=True)
if 'Index' in df.columns:
    df.drop(['Index'], axis=1, inplace=True)
if 'SMILES' in df.columns:
    df.drop(['SMILES'], axis=1, inplace=True)
if 'cmc' in df.columns:
    df.drop(['cmc'], axis=1, inplace=True)
elif 'Log(cmc*1000+1)' in df.columns:
    df.drop(['Log(cmc*1000+1)'], axis=1, inplace=True)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Hyperparameter range for num_iters
min_iters = 10
max_iters = 400
interval = 10

# Loop over the desired range of num_iters
for num_iters in range(min_iters, max_iters + 1, interval):
    print(f"Running PacMAP with num_iters = {num_iters}")
    
    # Dimensionality reduction using PacMAP with current num_iters
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0,
                              num_iters=num_iters, random_state=2021)  
    reduced_data = embedding.fit_transform(df_scaled)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=3, min_samples=5)
    cluster_labels = dbscan.fit_predict(reduced_data)
    
    # Get unique clusters and assign colors
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    # Create a scatter plot for visualization
    plt.figure(figsize=(8, 6))
    for cluster, color in zip(unique_clusters, colors):
        mask = cluster_labels == cluster
        label = f"Cluster {cluster}" if cluster != -1 else "Noise"
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], color=color, alpha=0.75, label=label)
    
    plt.xlabel("PaCMAP Component 1")
    plt.ylabel("PaCMAP Component 2")
    plt.title(f"PaCMAP Clustering using DBSCAN (num_iters={num_iters})")
    plt.legend(title="Clusters", loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot with the num_iters value in the filename
    plot_filename = os.path.join(results_folder, f"PaCMAP_Clustering_{num_iters}.png")
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to avoid memory issues in the loop

#make a gif of the images

from PIL import Image
import glob

# # Get a sorted list of image file paths (adjust the pattern if needed)
# image_files = sorted(glob.glob(f"{results_folder}\\*.png"))

# Get all image files and filter only those with iterations from 10 to 120
image_files = sorted(glob.glob(f"{results_folder}\\*.png"))

# Extract iteration numbers and filter within range
image_files = [file for file in image_files if 10 <= int(file.split("_")[-1].split(".")[0]) <= 400]

# Sort the filtered files numerically
image_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Open images and store them in a list
images = [Image.open(file) for file in image_files]

# Save the images as an animated GIF
# duration is in milliseconds per frame, loop=0 means infinite looping
images[0].save(f'{results_folder}\\GIF.gif', save_all=True, append_images=images[1:], duration=500, loop=0)

