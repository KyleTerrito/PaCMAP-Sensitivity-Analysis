import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import os
import pacmap
from sklearn.cluster import DBSCAN
import datetime
from sklearn.preprocessing import StandardScaler

'''MAKE SURE YOUR ORIGINAL DATA HAS A "Name" and "TARGET" COLUMN'''

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f"Results/{timestamp}"
os.makedirs(results_folder, exist_ok=True)
print(f"Created directory: {results_folder}")

# Load data
# file_path = os.path.join("DATA", "molecular_descriptors.csv")
# file_path = os.path.join("DATA", "molecular_descriptors_concat_data_product.csv")
# file_path = os.path.join("DATA", "molecular_descriptors_concat_data_product2_Target-Logcmc.csv")
file_path = os.path.join("DATA", "molecular_descriptors_concat_data_product2_Target-cmc.csv")
# file_path = os.path.join("DATA", "Data 6-16 Reduced.csv")
df = pd.read_csv(file_path)

############################################################################################################
# If the original dataset does not have a 'Name' column, create a random name for each row
def random_name():
    return ''.join(random.choices(string.ascii_uppercase, k=5))

if not 'Name' in df.columns:
    df['Name'] = [random_name() for _ in range(df.shape[0])]

# If the original dataset does not have a 'TARGET' column, create a random target value for each row
if not 'TARGET' in df.columns:
    df['TARGET'] = np.random.randint(0, 101, size=df.shape[0])  # Random integer target values

# If the original dataset does not have an 'Index' column, create an index column
if not 'Index' in df.columns:
    # df['Index'] = range(df.shape[0])  # Index column
    df['Index'] = range(1, df.shape[0]+ 1)  #start counting from 1

# Drop columns with SMILES data
cols_to_drop = [col for col in ['SMILES', 'smiles', 'Smiles'] if col in df.columns]
if cols_to_drop:
    df.drop(cols_to_drop, axis=1, inplace=True)
############################################################################################################

# Save the new dataset
output_path = os.path.join(results_folder, "original_data.csv")
df.to_csv(output_path, index=False)

# Store names and targets before dropping them
names = df['Name']
targets = df['TARGET']

names.to_csv(os.path.join(results_folder, "names.csv"), index=False)
targets.to_csv(os.path.join(results_folder, "targets.csv"), index=False)

# Drop non-feature columns
# df.drop(['Name', 'TARGET', 'Index'], axis=1, inplace=True)
columns_to_drop = ['Name', 'TARGET', 'Index']

if 'cmc' in df.columns:
    columns_to_drop.append('cmc')
elif 'Log(cmc*1000+1)' in df.columns:
    columns_to_drop.append('Log(cmc*1000+1)')

df.drop(columns=columns_to_drop, axis=1, inplace=True)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Reduce the data using PaCMAP
embedding = pacmap.PaCMAP(n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0, num_iters=100,random_state=101) #random_state=42)  
reduced_data = embedding.fit_transform(df_scaled)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=3, min_samples=5)  # Tune these parameters as needed
cluster_labels = dbscan.fit_predict(reduced_data)
cluster_labels_df = pd.DataFrame(cluster_labels, columns=["Cluster Labels"])
cluster_labels_df.to_csv(os.path.join(results_folder, "cluster_labels.csv"), index=False)
unique_clusters = np.unique(cluster_labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))


# Plot 1: PaCMAP visualization colored by TARGET values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=targets, cmap='viridis', alpha=0.75)
plt.colorbar(scatter, label='Target Values')
plt.xlabel("PaCMAP Component 1")
plt.ylabel("PaCMAP Component 2")
plt.title("PaCMAP DR (Colored by TARGET)")
plt.grid(True)

# Plot 2: PaCMAP visualization colored by DBSCAN cluster labels (with legend)
plt.subplot(1, 2, 2)

# Plot each cluster separately for a proper legend
for cluster, color in zip(unique_clusters, colors):
    mask = cluster_labels == cluster
    label = f"Cluster {cluster}" if cluster != -1 else "Noise"  # Label noise points properly
    plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], color=color, alpha=0.75, label=label)

plt.xlabel("PaCMAP Component 1")
plt.ylabel("PaCMAP Component 2")
plt.title("PaCMAP Clustering using DBSCAN")
# plt.legend(title="Clusters", loc="upper right", bbox_to_anchor=(1.2, 1))
plt.legend(title="Clusters", loc="center left", bbox_to_anchor=(1, .8))  # Legend for clusters
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "PaCMAP_Clustering.png"))

#save reduced data to results folder
reduced_data_df = pd.DataFrame(reduced_data, columns=["Dim1", "Dim2"])
# reduced_data_df["ID"] = range(reduced_data_df.shape[0])
reduced_data_df["ID"] = range(1, reduced_data_df.shape[0] + 1) #start counting from 1
reduced_data_df["Label"] = cluster_labels
reduced_data_df.to_csv(os.path.join(results_folder, "reduced_data.csv"), index=False)