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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from itertools import product
import time
import json

# Create NumPy-compatible JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

'''MAKE SURE YOUR ORIGINAL DATA HAS A "Name" and "TARGET" COLUMN'''

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f"Results/{timestamp}"
os.makedirs(results_folder, exist_ok=True)
print(f"Created directory: {results_folder}")

# Load data
file_path = os.path.join("DATA", "valid_molecules_data_cmc.csv")
df = pd.read_csv(file_path)

############################################################################################################
# If the original dataset does not have a 'TARGET' column, create a random target value for each row
if not 'TARGET' in df.columns:
    df['TARGET'] = np.random.randint(0, 101, size=df.shape[0])  # Random integer target values

# If the original dataset does not have an 'Index' column, create an index column
if not 'Index' in df.columns:
    df['Index'] = range(1, df.shape[0]+ 1)  #start counting from 1

# Drop columns with SMILES data
cols_to_drop = [col for col in ['SMILES', 'smiles', 'Smiles'] if col in df.columns]
if cols_to_drop:
    df.drop(cols_to_drop, axis=1, inplace=True)
############################################################################################################

# Save the new dataset
output_path = os.path.join(results_folder, "original_data.csv")
df.to_csv(output_path, index=False)

# Store targets before dropping them
targets = df['TARGET']
targets.to_csv(os.path.join(results_folder, "targets.csv"), index=False)

# Drop non-feature columns
columns_to_drop = ['TARGET', 'Index']

if 'cmc' in df.columns:
    columns_to_drop.append('cmc')
elif 'Log(cmc*1000+1)' in df.columns:
    columns_to_drop.append('Log(cmc*1000+1)')

df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Function to evaluate clustering quality
def evaluate_clustering(data, labels):
    # Skip evaluation if there's only one cluster or if all points are noise
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
        return {
            "silhouette_score": -1,
            "davies_bouldin_score": float('inf'),
            "calinski_harabasz_score": -1,
            "num_clusters": len(np.unique(labels[labels != -1])),
            "noise_ratio": np.sum(labels == -1) / len(labels)
        }
    
    try:
        sil_score = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else -1
    except:
        sil_score = -1
        
    try:
        db_score = davies_bouldin_score(data, labels) if len(np.unique(labels)) > 1 else float('inf')
    except:
        db_score = float('inf')
        
    try:
        ch_score = calinski_harabasz_score(data, labels) if len(np.unique(labels)) > 1 else -1
    except:
        ch_score = -1
    
    return {
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "calinski_harabasz_score": ch_score,
        "num_clusters": len(np.unique(labels[labels != -1])),
        "noise_ratio": np.sum(labels == -1) / len(labels)
    }

# PacMAP Hyperparameter Grid
pacmap_params = {
    'n_components': [2],  # Fixed for visualization
    'n_neighbors': [5, 10, 15, 20],
    'MN_ratio': [0.3, 0.5, 0.7],
    'FP_ratio': [1.0, 2.0, 3.0],
    'num_iters': [100, 200],  # Set iterations low for testing, increase for final runs
    'random_state': [101]  # Keep this fixed for reproducibility
}

# DBSCAN Hyperparameter Grid
dbscan_params = {
    'eps': [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],  # Increased eps values to create fewer, larger clusters
    'min_samples': [5, 10, 15, 20, 25, 30]  # Increased min_samples to require more points for clusters
}

# Optimization results storage
optimization_results = []

# Create a timestamp for logging
start_time = time.time()
print(f"Starting hyperparameter optimization at {datetime.datetime.now().strftime('%H:%M:%S')}")

# Generate all combinations of PacMAP params
pacmap_param_combinations = list(product(
    pacmap_params['n_neighbors'], 
    pacmap_params['MN_ratio'], 
    pacmap_params['FP_ratio'],
    pacmap_params['num_iters']
))

total_combinations = len(pacmap_param_combinations) * len(dbscan_params['eps']) * len(dbscan_params['min_samples'])
print(f"Testing {total_combinations} parameter combinations")

counter = 0
best_score = -float('inf')  # Initialize with worst possible score
best_params = None
best_reduced_data = None
best_labels = None

# For weighted scoring
weights = {
    'silhouette_score': 0.5,
    'davies_bouldin_score': -0.3,  # Negative because lower is better
    'calinski_harabasz_score': 0.2,
    'num_clusters': -0.3  # Penalty for too many clusters (negative because we want fewer clusters)
}

# Grid search through parameter combinations
for n_neighbors, MN_ratio, FP_ratio, num_iters in pacmap_param_combinations:
    # Apply PacMAP with current parameters
    embedding = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        num_iters=num_iters,
        random_state=101
    )
    
    reduced_data = embedding.fit_transform(df_scaled)
    
    # Test DBSCAN parameters with current PacMAP embedding
    for eps in dbscan_params['eps']:
        for min_samples in dbscan_params['min_samples']:
            counter += 1
            if counter % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {counter}/{total_combinations} combinations tested ({counter/total_combinations*100:.1f}%) - Elapsed time: {elapsed:.1f}s")
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(reduced_data)
            
            # Skip evaluation if all points are noise
            if np.all(cluster_labels == -1):
                continue
                
            # Evaluate clustering quality
            eval_metrics = evaluate_clustering(reduced_data, cluster_labels)
            
            # Calculate a weighted score
            num_clusters = eval_metrics['num_clusters']
            # Penalize having too many clusters (> 10) or too few clusters (< 3)
            cluster_penalty = 0
            if num_clusters > 10:
                cluster_penalty = (num_clusters - 10) * 0.1  # Penalty increases with more clusters
            elif num_clusters < 3:
                cluster_penalty = (3 - num_clusters) * 0.1  # Penalty for too few clusters
            
            weighted_score = (
                weights['silhouette_score'] * eval_metrics['silhouette_score'] +
                weights['davies_bouldin_score'] * (1 / (eval_metrics['davies_bouldin_score'] + 1e-5)) +  # Invert DB score as lower is better
                weights['calinski_harabasz_score'] * (eval_metrics['calinski_harabasz_score'] / 1000) -  # Scale CH score
                cluster_penalty  # Apply penalty for too many or too few clusters
            )
            
            # Store results
            result = {
                'pacmap_params': {
                    'n_neighbors': n_neighbors,
                    'MN_ratio': MN_ratio,
                    'FP_ratio': FP_ratio,
                    'num_iters': num_iters
                },
                'dbscan_params': {
                    'eps': eps,
                    'min_samples': min_samples
                },
                'metrics': eval_metrics,
                'weighted_score': weighted_score
            }
            optimization_results.append(result)
            
            # Update best parameters if better score is found
            if weighted_score > best_score:
                best_score = weighted_score
                best_params = {
                    'pacmap': {
                        'n_neighbors': n_neighbors,
                        'MN_ratio': MN_ratio,
                        'FP_ratio': FP_ratio,
                        'num_iters': num_iters
                    },
                    'dbscan': {
                        'eps': eps,
                        'min_samples': min_samples
                    }
                }
                best_reduced_data = reduced_data.copy()
                best_labels = cluster_labels.copy()

elapsed_time = time.time() - start_time
print(f"Optimization completed in {elapsed_time:.2f} seconds")

# Save optimization results
with open(os.path.join(results_folder, 'optimization_results.json'), 'w') as f:
    json.dump(optimization_results, f, cls=NumpyEncoder, indent=4)

# Sort results by weighted_score
sorted_results = sorted(optimization_results, key=lambda x: x['weighted_score'], reverse=True)
top_results = sorted_results[:10]

# Save top results to a more readable format
with open(os.path.join(results_folder, 'top_parameters.txt'), 'w') as f:
    f.write(f"Top {len(top_results)} Parameter Combinations:\n\n")
    for i, result in enumerate(top_results):
        f.write(f"Rank {i+1}:\n")
        f.write(f"  PacMAP Parameters:\n")
        for param, value in result['pacmap_params'].items():
            f.write(f"    {param}: {value}\n")
        f.write(f"  DBSCAN Parameters:\n")
        for param, value in result['dbscan_params'].items():
            f.write(f"    {param}: {value}\n")
        f.write(f"  Metrics:\n")
        for metric, value in result['metrics'].items():
            f.write(f"    {metric}: {value}\n")
        f.write(f"  Weighted Score: {result['weighted_score']:.4f}\n")
        f.write(f"  Number of Clusters: {result['metrics']['num_clusters']}\n\n")

print(f"Best parameters found:")
print(f"  PacMAP: {best_params['pacmap']}")
print(f"  DBSCAN: {best_params['dbscan']}")

# Re-run with best parameters to generate final results
print(f"Generating final visualizations with best parameters...")

# Apply best PacMAP parameters
final_embedding = pacmap.PaCMAP(
    n_components=2,
    n_neighbors=best_params['pacmap']['n_neighbors'],
    MN_ratio=best_params['pacmap']['MN_ratio'],
    FP_ratio=best_params['pacmap']['FP_ratio'],
    num_iters=best_params['pacmap']['num_iters'],
    random_state=101
)
final_reduced_data = final_embedding.fit_transform(df_scaled)

# Apply best DBSCAN parameters
final_dbscan = DBSCAN(
    eps=best_params['dbscan']['eps'],
    min_samples=best_params['dbscan']['min_samples']
)
final_cluster_labels = final_dbscan.fit_predict(final_reduced_data)

# Save final cluster labels
cluster_labels_df = pd.DataFrame(final_cluster_labels, columns=["Cluster Labels"])
cluster_labels_df.to_csv(os.path.join(results_folder, "cluster_labels.csv"), index=False)

# Get unique clusters for plotting
unique_clusters = np.unique(final_cluster_labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

# Create final visualizations
plt.figure(figsize=(15, 7.5))

# Plot 1: PaCMAP visualization colored by TARGET values
plt.subplot(1, 2, 1)
scatter = plt.scatter(final_reduced_data[:, 0], final_reduced_data[:, 1], c=targets, cmap='viridis', alpha=0.75)
cbar = plt.colorbar(scatter)
cbar.set_label('Log CMC values', fontsize=14)
cbar.ax.tick_params(labelsize=12)
plt.xlabel("PaCMAP Component 1", fontsize=14)
plt.ylabel("PaCMAP Component 2", fontsize=14)
plt.title(f"PaCMAP DR\nOptimal Parameters: n_neighbors={best_params['pacmap']['n_neighbors']}, MN={best_params['pacmap']['MN_ratio']}, FP={best_params['pacmap']['FP_ratio']}", fontsize=14)
plt.grid(True)

# Plot 2: PaCMAP visualization colored by DBSCAN cluster labels (with legend)
plt.subplot(1, 2, 2)

# Plot each cluster separately for a proper legend
for cluster, color in zip(unique_clusters, colors):
    mask = final_cluster_labels == cluster
    label = f"Cluster {cluster}" if cluster != -1 else "Noise"  # Label noise points properly
    plt.scatter(final_reduced_data[mask, 0], final_reduced_data[mask, 1], color=color, alpha=0.75, label=label)

plt.xlabel("PaCMAP Component 1", fontsize=14)
plt.ylabel("PaCMAP Component 2", fontsize=14)
plt.title(f"PaCMAP Clustering using DBSCAN\nOptimal Parameters: eps={best_params['dbscan']['eps']}, min_samples={best_params['dbscan']['min_samples']}", fontsize=14)
plt.legend(title="Clusters", loc="center left", bbox_to_anchor=(1, 0.8))
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "Optimized_PaCMAP_Clustering.png"), dpi=300, bbox_inches='tight')

# Create a visualization of the optimization process
plt.figure(figsize=(15, 10))

# 1. Plot silhouette scores vs. different parameter combinations
plt.subplot(2, 2, 1)
valid_results = [r for r in optimization_results if r['metrics']['silhouette_score'] > -1]
if valid_results:
    result_indices = range(len(valid_results))
    silhouette_scores = [r['metrics']['silhouette_score'] for r in valid_results]
    plt.bar(result_indices, silhouette_scores, alpha=0.7)
    plt.title('Silhouette Scores Across Parameter Combinations')
    plt.xlabel('Parameter Combination Index')
    plt.ylabel('Silhouette Score')
    plt.axhline(y=np.max(silhouette_scores), color='r', linestyle='-', label=f'Max: {np.max(silhouette_scores):.3f}')
    plt.legend()

# 2. Plot eps vs. number of clusters
plt.subplot(2, 2, 2)
eps_values = [r['dbscan_params']['eps'] for r in optimization_results]
num_clusters = [r['metrics']['num_clusters'] for r in optimization_results]
plt.scatter(eps_values, num_clusters, alpha=0.5)
plt.title('Effect of eps on Number of Clusters')
plt.xlabel('eps')
plt.ylabel('Number of Clusters')

# 3. Plot min_samples vs. noise ratio
plt.subplot(2, 2, 3)
min_samples_values = [r['dbscan_params']['min_samples'] for r in optimization_results]
noise_ratios = [r['metrics']['noise_ratio'] for r in optimization_results]
plt.scatter(min_samples_values, noise_ratios, alpha=0.5)
plt.title('Effect of min_samples on Noise Ratio')
plt.xlabel('min_samples')
plt.ylabel('Noise Ratio')

# 4. Plot top 10 weighted scores
plt.subplot(2, 2, 4)
top_indices = range(min(10, len(sorted_results)))
top_scores = [sorted_results[i]['weighted_score'] for i in top_indices]
plt.bar(top_indices, top_scores)
plt.title('Top 10 Parameter Combinations by Weighted Score')
plt.xlabel('Rank')
plt.ylabel('Weighted Score')

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "Optimization_Analysis.png"), dpi=300)

# Save reduced data to results folder
reduced_data_df = pd.DataFrame(final_reduced_data, columns=["Dim1", "Dim2"])
reduced_data_df["ID"] = range(1, reduced_data_df.shape[0] + 1)  # start counting from 1
reduced_data_df["Label"] = final_cluster_labels
reduced_data_df["Target"] = targets.values
reduced_data_df.to_csv(os.path.join(results_folder, "reduced_data.csv"), index=False)

# Save best parameters
with open(os.path.join(results_folder, 'best_parameters.json'), 'w') as f:
    json.dump(best_params, f, cls=NumpyEncoder, indent=4)

print(f"All results saved to {results_folder}")
print(f"Total runtime: {time.time() - start_time:.2f} seconds")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import string
# import os
# import pacmap
# from sklearn.cluster import DBSCAN
# import datetime
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# from itertools import product
# import time
# import json

# '''MAKE SURE YOUR ORIGINAL DATA HAS A "Name" and "TARGET" COLUMN'''

# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# results_folder = f"Results/{timestamp}"
# os.makedirs(results_folder, exist_ok=True)
# print(f"Created directory: {results_folder}")

# # Load data
# file_path = os.path.join("DATA", "valid_molecules_data_cmc.csv")
# df = pd.read_csv(file_path)

# ############################################################################################################
# # If the original dataset does not have a 'TARGET' column, create a random target value for each row
# if not 'TARGET' in df.columns:
#     df['TARGET'] = np.random.randint(0, 101, size=df.shape[0])  # Random integer target values

# # If the original dataset does not have an 'Index' column, create an index column
# if not 'Index' in df.columns:
#     df['Index'] = range(1, df.shape[0]+ 1)  #start counting from 1

# # Drop columns with SMILES data
# cols_to_drop = [col for col in ['SMILES', 'smiles', 'Smiles'] if col in df.columns]
# if cols_to_drop:
#     df.drop(cols_to_drop, axis=1, inplace=True)
# ############################################################################################################

# # Save the new dataset
# output_path = os.path.join(results_folder, "original_data.csv")
# df.to_csv(output_path, index=False)

# # Store targets before dropping them
# targets = df['TARGET']
# targets.to_csv(os.path.join(results_folder, "targets.csv"), index=False)

# # Drop non-feature columns
# columns_to_drop = ['TARGET', 'Index']

# if 'cmc' in df.columns:
#     columns_to_drop.append('cmc')
# elif 'Log(cmc*1000+1)' in df.columns:
#     columns_to_drop.append('Log(cmc*1000+1)')

# df.drop(columns=columns_to_drop, axis=1, inplace=True)

# # Scale the data
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df)

# # Function to evaluate clustering quality
# def evaluate_clustering(data, labels):
#     # Skip evaluation if there's only one cluster or if all points are noise
#     unique_labels = np.unique(labels)
#     if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
#         return {
#             "silhouette_score": -1,
#             "davies_bouldin_score": float('inf'),
#             "calinski_harabasz_score": -1,
#             "num_clusters": len(np.unique(labels[labels != -1])),
#             "noise_ratio": np.sum(labels == -1) / len(labels)
#         }
    
#     try:
#         sil_score = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else -1
#     except:
#         sil_score = -1
        
#     try:
#         db_score = davies_bouldin_score(data, labels) if len(np.unique(labels)) > 1 else float('inf')
#     except:
#         db_score = float('inf')
        
#     try:
#         ch_score = calinski_harabasz_score(data, labels) if len(np.unique(labels)) > 1 else -1
#     except:
#         ch_score = -1
    
#     return {
#         "silhouette_score": sil_score,
#         "davies_bouldin_score": db_score,
#         "calinski_harabasz_score": ch_score,
#         "num_clusters": len(np.unique(labels[labels != -1])),
#         "noise_ratio": np.sum(labels == -1) / len(labels)
#     }

# # PacMAP Hyperparameter Grid
# pacmap_params = {
#     'n_components': [2],  # Fixed for visualization
#     'n_neighbors': [5, 10, 15, 20],
#     'MN_ratio': [0.3, 0.5, 0.7],
#     'FP_ratio': [1.0, 2.0, 3.0],
#     'num_iters': [100, 200],  # Set iterations low for testing, increase for final runs
#     'random_state': [101]  # Keep this fixed for reproducibility
# }

# # DBSCAN Hyperparameter Grid
# dbscan_params = {
#     'eps': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
#     'min_samples': [3, 5, 10, 15, 20]
# }

# # Optimization results storage
# optimization_results = []

# # Create a timestamp for logging
# start_time = time.time()
# print(f"Starting hyperparameter optimization at {datetime.datetime.now().strftime('%H:%M:%S')}")

# # Generate all combinations of PacMAP params
# pacmap_param_combinations = list(product(
#     pacmap_params['n_neighbors'], 
#     pacmap_params['MN_ratio'], 
#     pacmap_params['FP_ratio'],
#     pacmap_params['num_iters']
# ))

# total_combinations = len(pacmap_param_combinations) * len(dbscan_params['eps']) * len(dbscan_params['min_samples'])
# print(f"Testing {total_combinations} parameter combinations")

# counter = 0
# best_score = -float('inf')  # Initialize with worst possible score
# best_params = None
# best_reduced_data = None
# best_labels = None

# # For weighted scoring
# weights = {
#     'silhouette_score': 0.5,
#     'davies_bouldin_score': -0.3,  # Negative because lower is better
#     'calinski_harabasz_score': 0.2
# }

# # Grid search through parameter combinations
# for n_neighbors, MN_ratio, FP_ratio, num_iters in pacmap_param_combinations:
#     # Apply PacMAP with current parameters
#     embedding = pacmap.PaCMAP(
#         n_components=2,
#         n_neighbors=n_neighbors,
#         MN_ratio=MN_ratio,
#         FP_ratio=FP_ratio,
#         num_iters=num_iters,
#         random_state=101
#     )
    
#     reduced_data = embedding.fit_transform(df_scaled)
    
#     # Test DBSCAN parameters with current PacMAP embedding
#     for eps in dbscan_params['eps']:
#         for min_samples in dbscan_params['min_samples']:
#             counter += 1
#             if counter % 10 == 0:
#                 elapsed = time.time() - start_time
#                 print(f"Progress: {counter}/{total_combinations} combinations tested ({counter/total_combinations*100:.1f}%) - Elapsed time: {elapsed:.1f}s")
            
#             # Apply DBSCAN
#             dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#             cluster_labels = dbscan.fit_predict(reduced_data)
            
#             # Skip evaluation if all points are noise
#             if np.all(cluster_labels == -1):
#                 continue
                
#             # Evaluate clustering quality
#             eval_metrics = evaluate_clustering(reduced_data, cluster_labels)
            
#             # Calculate a weighted score
#             weighted_score = (
#                 weights['silhouette_score'] * eval_metrics['silhouette_score'] +
#                 weights['davies_bouldin_score'] * (1 / (eval_metrics['davies_bouldin_score'] + 1e-5)) +  # Invert DB score as lower is better
#                 weights['calinski_harabasz_score'] * (eval_metrics['calinski_harabasz_score'] / 1000)  # Scale CH score
#             )
            
#             # Store results
#             result = {
#                 'pacmap_params': {
#                     'n_neighbors': n_neighbors,
#                     'MN_ratio': MN_ratio,
#                     'FP_ratio': FP_ratio,
#                     'num_iters': num_iters
#                 },
#                 'dbscan_params': {
#                     'eps': eps,
#                     'min_samples': min_samples
#                 },
#                 'metrics': eval_metrics,
#                 'weighted_score': weighted_score
#             }
#             optimization_results.append(result)
            
#             # Update best parameters if better score is found
#             if weighted_score > best_score:
#                 best_score = weighted_score
#                 best_params = {
#                     'pacmap': {
#                         'n_neighbors': n_neighbors,
#                         'MN_ratio': MN_ratio,
#                         'FP_ratio': FP_ratio,
#                         'num_iters': num_iters
#                     },
#                     'dbscan': {
#                         'eps': eps,
#                         'min_samples': min_samples
#                     }
#                 }
#                 best_reduced_data = reduced_data.copy()
#                 best_labels = cluster_labels.copy()

# elapsed_time = time.time() - start_time
# print(f"Optimization completed in {elapsed_time:.2f} seconds")

# # Save optimization results
# with open(os.path.join(results_folder, 'optimization_results.json'), 'w') as f:
#     json.dump(optimization_results, f, indent=4)

# # Sort results by weighted_score
# sorted_results = sorted(optimization_results, key=lambda x: x['weighted_score'], reverse=True)
# top_results = sorted_results[:10]

# # Save top results to a more readable format
# with open(os.path.join(results_folder, 'top_parameters.txt'), 'w') as f:
#     f.write(f"Top {len(top_results)} Parameter Combinations:\n\n")
#     for i, result in enumerate(top_results):
#         f.write(f"Rank {i+1}:\n")
#         f.write(f"  PacMAP Parameters:\n")
#         for param, value in result['pacmap_params'].items():
#             f.write(f"    {param}: {value}\n")
#         f.write(f"  DBSCAN Parameters:\n")
#         for param, value in result['dbscan_params'].items():
#             f.write(f"    {param}: {value}\n")
#         f.write(f"  Metrics:\n")
#         for metric, value in result['metrics'].items():
#             f.write(f"    {metric}: {value}\n")
#         f.write(f"  Weighted Score: {result['weighted_score']:.4f}\n\n")

# print(f"Best parameters found:")
# print(f"  PacMAP: {best_params['pacmap']}")
# print(f"  DBSCAN: {best_params['dbscan']}")

# # Re-run with best parameters to generate final results
# print(f"Generating final visualizations with best parameters...")

# # Apply best PacMAP parameters
# final_embedding = pacmap.PaCMAP(
#     n_components=2,
#     n_neighbors=best_params['pacmap']['n_neighbors'],
#     MN_ratio=best_params['pacmap']['MN_ratio'],
#     FP_ratio=best_params['pacmap']['FP_ratio'],
#     num_iters=best_params['pacmap']['num_iters'],
#     random_state=101
# )
# final_reduced_data = final_embedding.fit_transform(df_scaled)

# # Apply best DBSCAN parameters
# final_dbscan = DBSCAN(
#     eps=best_params['dbscan']['eps'],
#     min_samples=best_params['dbscan']['min_samples']
# )
# final_cluster_labels = final_dbscan.fit_predict(final_reduced_data)

# # Save final cluster labels
# cluster_labels_df = pd.DataFrame(final_cluster_labels, columns=["Cluster Labels"])
# cluster_labels_df.to_csv(os.path.join(results_folder, "cluster_labels.csv"), index=False)

# # Get unique clusters for plotting
# unique_clusters = np.unique(final_cluster_labels)
# colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

# # Create final visualizations
# plt.figure(figsize=(15, 7.5))

# # Plot 1: PaCMAP visualization colored by TARGET values
# plt.subplot(1, 2, 1)
# scatter = plt.scatter(final_reduced_data[:, 0], final_reduced_data[:, 1], c=targets, cmap='viridis', alpha=0.75)
# plt.colorbar(scatter, label='Target Values')
# plt.xlabel("PaCMAP Component 1")
# plt.ylabel("PaCMAP Component 2")
# plt.title(f"PaCMAP DR (Colored by TARGET)\nOptimal Parameters: n_neighbors={best_params['pacmap']['n_neighbors']}, MN={best_params['pacmap']['MN_ratio']}, FP={best_params['pacmap']['FP_ratio']}")
# plt.grid(True)

# # Plot 2: PaCMAP visualization colored by DBSCAN cluster labels (with legend)
# plt.subplot(1, 2, 2)

# # Plot each cluster separately for a proper legend
# for cluster, color in zip(unique_clusters, colors):
#     mask = final_cluster_labels == cluster
#     label = f"Cluster {cluster}" if cluster != -1 else "Noise"  # Label noise points properly
#     plt.scatter(final_reduced_data[mask, 0], final_reduced_data[mask, 1], color=color, alpha=0.75, label=label)

# plt.xlabel("PaCMAP Component 1")
# plt.ylabel("PaCMAP Component 2")
# plt.title(f"PaCMAP Clustering using DBSCAN\nOptimal Parameters: eps={best_params['dbscan']['eps']}, min_samples={best_params['dbscan']['min_samples']}")
# plt.legend(title="Clusters", loc="center left", bbox_to_anchor=(1, 0.8))
# plt.grid(True)

# plt.tight_layout()
# plt.savefig(os.path.join(results_folder, "Optimized_PaCMAP_Clustering.png"), dpi=300, bbox_inches='tight')

# # Create a visualization of the optimization process
# plt.figure(figsize=(15, 10))

# # 1. Plot silhouette scores vs. different parameter combinations
# plt.subplot(2, 2, 1)
# valid_results = [r for r in optimization_results if r['metrics']['silhouette_score'] > -1]
# if valid_results:
#     result_indices = range(len(valid_results))
#     silhouette_scores = [r['metrics']['silhouette_score'] for r in valid_results]
#     plt.bar(result_indices, silhouette_scores, alpha=0.7)
#     plt.title('Silhouette Scores Across Parameter Combinations')
#     plt.xlabel('Parameter Combination Index')
#     plt.ylabel('Silhouette Score')
#     plt.axhline(y=np.max(silhouette_scores), color='r', linestyle='-', label=f'Max: {np.max(silhouette_scores):.3f}')
#     plt.legend()

# # 2. Plot eps vs. number of clusters
# plt.subplot(2, 2, 2)
# eps_values = [r['dbscan_params']['eps'] for r in optimization_results]
# num_clusters = [r['metrics']['num_clusters'] for r in optimization_results]
# plt.scatter(eps_values, num_clusters, alpha=0.5)
# plt.title('Effect of eps on Number of Clusters')
# plt.xlabel('eps')
# plt.ylabel('Number of Clusters')

# # 3. Plot min_samples vs. noise ratio
# plt.subplot(2, 2, 3)
# min_samples_values = [r['dbscan_params']['min_samples'] for r in optimization_results]
# noise_ratios = [r['metrics']['noise_ratio'] for r in optimization_results]
# plt.scatter(min_samples_values, noise_ratios, alpha=0.5)
# plt.title('Effect of min_samples on Noise Ratio')
# plt.xlabel('min_samples')
# plt.ylabel('Noise Ratio')

# # 4. Plot top 10 weighted scores
# plt.subplot(2, 2, 4)
# top_indices = range(min(10, len(sorted_results)))
# top_scores = [sorted_results[i]['weighted_score'] for i in top_indices]
# plt.bar(top_indices, top_scores)
# plt.title('Top 10 Parameter Combinations by Weighted Score')
# plt.xlabel('Rank')
# plt.ylabel('Weighted Score')

# plt.tight_layout()
# plt.savefig(os.path.join(results_folder, "Optimization_Analysis.png"), dpi=300)

# # Save reduced data to results folder
# reduced_data_df = pd.DataFrame(final_reduced_data, columns=["Dim1", "Dim2"])
# reduced_data_df["ID"] = range(1, reduced_data_df.shape[0] + 1)  # start counting from 1
# reduced_data_df["Label"] = final_cluster_labels
# reduced_data_df["Target"] = targets.values
# reduced_data_df.to_csv(os.path.join(results_folder, "reduced_data.csv"), index=False)

# # Save best parameters
# with open(os.path.join(results_folder, 'best_parameters.json'), 'w') as f:
#     json.dump(best_params, f, indent=4)

# print(f"All results saved to {results_folder}")
# print(f"Total runtime: {time.time() - start_time:.2f} seconds")