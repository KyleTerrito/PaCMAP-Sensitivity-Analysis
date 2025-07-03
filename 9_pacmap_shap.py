import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pacmap
from sklearn.cluster import DBSCAN
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import os
import datetime

def main():
    # Create results directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_name = 'data1'
    results_folder = f"Results/{data_name}_DR_cluster_shap/{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    print(f"Created directory: {results_folder}")
    
    # Data loading
    # data_name = 'data1'
    if data_name == 'data1':
        file_path = os.path.join("DATA", "valid_molecules_data_cmc.csv")
        df_data1 = pd.read_csv(file_path)
        df_data1_copy = df_data1.copy()
    else:
        file_path = os.path.join("DATA", "valid_molecules_data_cmc_data2.csv")
        df_data1 = pd.read_csv(file_path)
        df_data1_copy = df_data1.copy()
    
    # Data preprocessing
    if 'smiles' in df_data1.columns:
        df_data1 = df_data1.drop(columns=['smiles'])
    if 'TARGET' in df_data1.columns:
        y_target = df_data1_copy['TARGET']
        df_data1 = df_data1.drop(columns=['TARGET'])
    
    print(f"Data shape: {df_data1.shape}")
    
    # Scale data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_data1)
    
    # PaCMAP dimensionality reduction
    pacmap_reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.6, 
                                   FP_ratio=1.5, num_iters=200, random_state=101)
    X_pacmap = pacmap_reducer.fit_transform(df_scaled)
    
    # Plot PaCMAP with TARGET coloring
    plt.figure(figsize=(5, 6))
    scatter = plt.scatter(X_pacmap[:, 0], X_pacmap[:, 1], c=df_data1_copy['TARGET'], cmap="viridis")
    plt.colorbar(scatter, label='TARGET')
    plt.title("PaCMAP Projection with TARGET")
    plt.savefig(os.path.join(results_folder, "pacmap_target.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=3, min_samples=5)
    cluster_labels = dbscan.fit_predict(X_pacmap)
    
    # Plot PaCMAP with cluster labels
    plt.figure(figsize=(5, 6))
    scatter = plt.scatter(X_pacmap[:, 0], X_pacmap[:, 1], c=cluster_labels, cmap="tab10", s=30, alpha=1)
    
    # Create a custom legend
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.tab10(unique_clusters / max(1, max(unique_clusters)))  # Prevent division by zero
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, 
                                  label=f'Cluster {cluster}')
                      for cluster, color in zip(unique_clusters, colors)]
    plt.legend(handles=legend_elements, title="Clusters", loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    
    plt.title(f"PaCMAP Clustering using DBSCAN \n{data_name}", fontsize=14)
    plt.xlabel("PaCMAP Component 1", fontsize=14)
    plt.ylabel("PaCMAP Component 2", fontsize=14)
    plt.savefig(os.path.join(results_folder, "pacmap_clusters.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # XGBoost model training
    X = df_data1  # Features
    y = cluster_labels  # Target (cluster labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model evaluation (Accuracy):", accuracy)
    
    # Save model results
    with open(os.path.join(results_folder, "model_results.txt"), "w") as f:
        f.write(f"Model: XGBoost\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Data shape: {df_data1.shape}\n")
        f.write(f"Number of clusters: {len(unique_clusters)}\n")
    
    # SHAP analysis
    explainer = shap.Explainer(xgb_model, X_train)
    shap_values = explainer(X_test)
    
    # Generate SHAP plots for each cluster
    for cluster in unique_clusters:
        print(f"Generating SHAP plot for Cluster {cluster}")
        cluster_indices = np.where(y_test == cluster)[0]
        
        if len(cluster_indices) > 0:
            class_preds = xgb_model.predict(X_test.iloc[cluster_indices])
            n_samples = len(cluster_indices)
            n_features = X_test.shape[1]
            
            filtered_values = np.zeros((n_samples, n_features))
            
            for i, sample_idx in enumerate(cluster_indices):
                pred_class = class_preds[i]
                shap_obj = shap_values[sample_idx]
                filtered_values[i, :] = shap_obj.values[:, pred_class]
            
            X_cluster = X_test.iloc[cluster_indices]
            
            # Plot clean bar plot
            plt.figure(figsize=(8, 6))
            shap.summary_plot(filtered_values, X_cluster, plot_type="bar", show=False)
            plt.title(f"Top Feature Importances for Cluster {cluster} \n{data_name}", fontsize=14)
            plt.gcf().set_size_inches(8, 6)
            plt.subplots_adjust(left=0.35)  # Adjust left margin for long feature names
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, f"shap_cluster_{cluster}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print(f"No data points in Cluster {cluster} for the test set.")
    
    # Overall SHAP summary plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        max_display=10,
        plot_type="bar",
        show=False
    )
    
    # Adjust legend labels from 'Class' to 'Cluster'
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [label.replace("Class", "Cluster") for label in labels]
    plt.legend(handles, new_labels, title="Cluster", fontsize=10, title_fontsize=11)
    
    plt.gcf().set_size_inches(9, 6)
    plt.tight_layout()
    plt.title(f"Top 10 Feature Importances by Cluster\n {data_name}", fontsize=14)
    plt.savefig(os.path.join(results_folder, "shap_overall.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis complete. Results saved to {results_folder}")

if __name__ == "__main__":
    main()