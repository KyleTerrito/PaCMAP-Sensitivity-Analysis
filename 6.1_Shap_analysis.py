import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def create_stacked_perm_importance_plot(perm_importance_results, feature_names, output_folder, num_classes, top_n=10):
    """
    Create a stacked horizontal bar chart showing permutation importance by cluster.
    
    Parameters:
    perm_importance_results (dict): Dictionary of permutation importance results by cluster
    feature_names (list): List of feature names
    output_folder (str): Path to save output files
    num_classes (int): Number of clusters
    top_n (int): Number of top features to display
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Calculate importance values for each feature and cluster
    importance_matrix = np.zeros((len(feature_names), num_classes))
    for cluster_idx in range(num_classes):
        importance_matrix[:, cluster_idx] = perm_importance_results[cluster_idx].importances_mean
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame(importance_matrix, index=feature_names)
    
    # Get total importance and sort features
    importance_df['total'] = importance_df.sum(axis=1)
    importance_df = importance_df.sort_values('total', ascending=False)
    importance_df = importance_df.drop('total', axis=1)
    
    # Get top N features
    top_features = importance_df.head(top_n).index.tolist()
    top_importance_df = importance_df.loc[top_features]
    
    # Reverse to have most important at the top
    top_importance_df = top_importance_df.iloc[::-1]

    # Use tab10 colormap to match the cluster visualization
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    # # Create color map for clusters
    # colors = plt.cm.get_cmap( 'Set2', num_classes)
    # cluster_colors = [colors(i) for i in range(num_classes)]
    
    # Create stacked bar chart
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    left = np.zeros(len(top_features))
    for cluster_idx in range(num_classes):
        ax.barh(
            top_features, 
            top_importance_df.iloc[:, cluster_idx], 
            left=left, 
            # color=cluster_colors[cluster_idx],
            color=colors[cluster_idx],
            label=f'Cluster {cluster_idx}'
        )
        left += top_importance_df.iloc[:, cluster_idx]
    
    # Add labels and legend
    ax.set_xlabel('Permutation Importance', fontsize=14)
    ax.set_title('Feature Importance by Cluster (Permutation Importance)', fontsize=14)
    
    # Adjust y-axis (features) font size
    ax.tick_params(axis='y', labelsize=12)

    # If you also want to adjust x-axis tick labels
    ax.tick_params(axis='x', labelsize=12)  

    # Create legend patches
    # patches = [mpatches.Patch(color=colors(i), label=f'Cluster {i}') for i in range(num_classes)]
    patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i}') for i in range(num_classes)]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "stacked_permutation_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the data
    top_importance_df.to_csv(os.path.join(output_folder, "stacked_permutation_importance.csv"))
    
    print(f"Stacked permutation importance plot saved to {output_folder}")

# def create_stacked_perm_importance_plot(perm_importance_results, feature_names, output_folder, num_classes, top_n=10):
#     """
#     Create a stacked horizontal bar chart showing permutation importance by cluster.
#     Using colors similar to those in SHAP visualizations.
#     """
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
    
#     # Calculate importance values for each feature and cluster
#     importance_matrix = np.zeros((len(feature_names), num_classes))
#     for cluster_idx in range(num_classes):
#         importance_matrix[:, cluster_idx] = perm_importance_results[cluster_idx].importances_mean
    
#     # Create DataFrame for plotting
#     importance_df = pd.DataFrame(importance_matrix, index=feature_names)
    
#     # Get total importance and sort features
#     importance_df['total'] = importance_df.sum(axis=1)
#     importance_df = importance_df.sort_values('total', ascending=False)
#     importance_df = importance_df.drop('total', axis=1)
    
#     # Get top N features
#     top_features = importance_df.head(top_n).index.tolist()
#     top_importance_df = importance_df.loc[top_features]
    
#     # Reverse to have most important at the top
#     top_importance_df = top_importance_df.iloc[::-1]
    
#     # SHAP-like colors (approximating colors often used in SHAP categorical plots)
#     # This color scheme is derived from the default matplotlib color cycle, which is similar 
#     # to what SHAP uses for categorical data
#     shap_colors = [
#         '#1f77b4',  # blue
#         '#ff7f0e',  # orange
#         '#2ca02c',  # green
#         '#d62728',  # red
#         '#9467bd',  # purple
#         '#8c564b',  # brown
#         '#e377c2',  # pink
#         '#7f7f7f',  # gray
#         '#bcbd22',  # olive
#         '#17becf'   # cyan
#     ]
    
#     # Create stacked bar chart
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111)
    
#     left = np.zeros(len(top_features))
#     for cluster_idx in range(num_classes):
#         ax.barh(
#             top_features, 
#             top_importance_df.iloc[:, cluster_idx], 
#             left=left, 
#             color=shap_colors[cluster_idx % len(shap_colors)], 
#             label=f'Cluster {cluster_idx}'
#         )
#         left += top_importance_df.iloc[:, cluster_idx]
    
#     # Add labels and legend
#     ax.set_xlabel('Permutation Importance')
#     ax.set_title('Feature Importance by Cluster (Permutation Importance)')
    
#     # Create legend patches
#     patches = [mpatches.Patch(color=shap_colors[i % len(shap_colors)], label=f'Cluster {i}') 
#                for i in range(num_classes)]
#     ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, "stacked_permutation_importance.png"), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Save the data
#     top_importance_df.to_csv(os.path.join(output_folder, "stacked_permutation_importance.csv"))
    
#     print(f"Stacked permutation importance plot saved to {output_folder}")

def analyze_shap_contributions(input_folder, output_folder=None):
    """
    Analyze feature contributions for cluster classification using XGBoost and SHAP.
    
    Parameters:
    input_folder (str): Path to the folder containing the results files
    output_folder (str, optional): Path to save the output files. If None, creates a new folder
    
    Returns:
    str: Path to the output folder
    """
    # Create output folder if not provided
    if output_folder is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = f"Results_SHAP/Cluster_Analysis_{timestamp}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created directory: {output_folder}")
    
    # Load the data
    original_data = pd.read_csv(os.path.join(input_folder, "original_data.csv"))
    cluster_labels = pd.read_csv(os.path.join(input_folder, "cluster_labels.csv"))["Cluster Labels"]

    # Filter out noise (-1) labels
    valid_indices = cluster_labels != -1
    cluster_labels = cluster_labels[valid_indices]
    original_data = original_data[valid_indices]
    
    # Extract feature columns (drop non-feature columns)
    columns_to_drop = []
    for col in ['Name', 'TARGET', 'Index', 'cmc']:
        if col in original_data.columns:
            columns_to_drop.append(col)
    
    # Get feature data
    feature_data = original_data.copy()
    feature_data.drop(columns=columns_to_drop, axis=1, inplace=True)
    feature_columns = feature_data.columns.tolist()
    
    # Prepare data for XGBoost
    X = feature_data.values
    y = cluster_labels.values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Number of clusters: {len(np.unique(y))}")
    cluster_counts = pd.Series(y).value_counts().sort_index()
    print("Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples")
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    
    # Set up XGBoost parameters for multiclass classification
    num_classes = len(np.unique(y))
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Create and train the model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print classification report
    class_report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(class_report)
    
    # Save classification report
    with open(os.path.join(output_folder, "classification_report.txt"), "w") as f:
        f.write(f"Model accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
    
    # Create and save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[f"Cluster {i}" for i in range(num_classes)],
               yticklabels=[f"Cluster {i}" for i in range(num_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.close()
    
    # Feature importance analysis using built-in XGBoost feature importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=15, height=0.8)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "xgboost_feature_importance.png"))
    plt.close()
    
    # Save feature importance scores
    feature_importance = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv(os.path.join(output_folder, "feature_importance.csv"), index=False)
    
    # SHAP Analysis
    print("\nPerforming SHAP analysis...")
    
    try:
        # Use KernelExplainer instead of TreeExplainer to avoid shape issues
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        
        # Use a sample of the test set for SHAP values to reduce computation time
        sample_size = min(50, X_test.shape[0])  # Reduced sample size for faster computation
        X_sample = X_test[:sample_size]
        
        # Calculate SHAP values (this may take time with KernelExplainer)
        print("Calculating SHAP values (this may take a few minutes)...")
        shap_values = explainer.shap_values(X_sample)
        
        # Global SHAP summary plot for all clusters combined
        plt.figure(figsize=(12, 10))
        # First element corresponds to cluster 0
        shap.summary_plot(shap_values[0], X_sample, feature_names=feature_columns, show=False)
        plt.title("SHAP Summary Plot (Cluster 0)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "shap_summary_cluster_0.png"))
        plt.close()
        
        # Individual summary plots for each cluster
        for cluster_idx in range(num_classes):
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values[cluster_idx], X_sample, feature_names=feature_columns, show=False)
            plt.title(f"SHAP Summary for Cluster {cluster_idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"shap_summary_cluster_{cluster_idx}.png"))
            plt.close()
        
        # Pairwise comparison between clusters
        print("\nAnalyzing pairwise differences between clusters...")
        cluster_pairs = []
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                cluster_pairs.append((i, j))
            
        # For each cluster pair, find the top differentiating features
        for cluster1, cluster2 in cluster_pairs:
            # Calculate absolute difference between SHAP values for the two clusters
            shap_diff = np.abs(np.array(shap_values[cluster1]) - np.array(shap_values[cluster2]))
            
            # Calculate mean absolute difference for each feature
            mean_shap_diff = np.mean(shap_diff, axis=0)
            
            # Create a DataFrame with features and their SHAP difference
            diff_df = pd.DataFrame({
                'Feature': feature_columns,
                'SHAP_Difference': mean_shap_diff
            }).sort_values('SHAP_Difference', ascending=False)
            
            # Save to CSV
            diff_df.to_csv(os.path.join(output_folder, f"shap_diff_cluster_{cluster1}_vs_{cluster2}.csv"), index=False)
            
            # Plot top differentiating features
            top_n = 10
            plt.figure(figsize=(10, 6))
            sns.barplot(x='SHAP_Difference', y='Feature', data=diff_df.head(top_n), palette='viridis')
            plt.title(f"Top Features Differentiating Cluster {cluster1} vs Cluster {cluster2}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"shap_diff_cluster_{cluster1}_vs_{cluster2}.png"))
            plt.close()
            
            # Print top 5 differentiating features
            print(f"\nTop differentiating features between Cluster {cluster1} and Cluster {cluster2}:")
            for i, row in diff_df.head(5).iterrows():
                print(f"  {row['Feature']}: {row['SHAP_Difference']:.4f}")
    
        # Create a combined feature importance heatmap
        # Extract mean absolute SHAP values per feature per cluster
        shap_importance = np.zeros((len(feature_columns), num_classes))
        for cluster_idx in range(num_classes):
            shap_importance[:, cluster_idx] = np.abs(shap_values[cluster_idx]).mean(axis=0)
        
        # Create a DataFrame for the heatmap
        heatmap_df = pd.DataFrame(
            shap_importance, 
            index=feature_columns, 
            columns=[f"Cluster {i}" for i in range(num_classes)]
        )
        
        # Sort features by overall importance
        total_importance = heatmap_df.sum(axis=1).sort_values(ascending=False)
        heatmap_df = heatmap_df.loc[total_importance.index]
        
        # Plot the heatmap
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(heatmap_df.head(15), cmap="YlGnBu", annot=True, fmt=".3f")
        plt.title("Feature Importance Across Clusters (Mean |SHAP| values)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "feature_importance_heatmap.png"))
        plt.close()
        
        # Save the full heatmap data
        heatmap_df.to_csv(os.path.join(output_folder, "feature_importance_by_cluster.csv"))
    
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        print("Falling back to alternative approach...")
        
        # Alternative approach using permutation importance
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance for each cluster
        print("Calculating permutation importance for each cluster...")
        
        perm_importance_results = {}
        for cluster_idx in range(num_classes):
            # Convert to binary classification problem (this cluster vs. others)
            y_train_binary = (y_train == cluster_idx).astype(int)
            y_test_binary = (y_test == cluster_idx).astype(int)
            
            # Create and train a binary classifier
            binary_model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=4,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            binary_model.fit(X_train, y_train_binary)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                binary_model, X_test, y_test_binary, 
                n_repeats=10, random_state=42
            )
            
            # Store results
            perm_importance_results[cluster_idx] = perm_importance
            
            # Create and save plot for this cluster
            sorted_idx = perm_importance.importances_mean.argsort()[-15:]
            plt.figure(figsize=(10, 8))
            plt.barh(
                [feature_columns[i] for i in sorted_idx], 
                perm_importance.importances_mean[sorted_idx]
            )
            plt.xlabel("Permutation Importance")
            plt.title(f"Feature Importance for Cluster {cluster_idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"permutation_importance_cluster_{cluster_idx}.png"))
            plt.close()
            
            # Save to CSV
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            }).sort_values('Importance', ascending=False)
            
            importance_df.to_csv(
                os.path.join(output_folder, f"permutation_importance_cluster_{cluster_idx}.csv"), 
                index=False
            )
        
        # Create stacked permutation importance plot
        print("Creating stacked permutation importance plot...")
        create_stacked_perm_importance_plot(
            perm_importance_results,
            feature_columns,
            output_folder,
            num_classes,
            top_n=10
        )

        # Create pairwise comparison of feature importance
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                # Calculate difference in feature importance between two clusters
                imp_i = perm_importance_results[i].importances_mean
                imp_j = perm_importance_results[j].importances_mean
                
                # Absolute difference
                abs_diff = np.abs(imp_i - imp_j)
                
                # Create DataFrame and sort
                diff_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance_Difference': abs_diff
                }).sort_values('Importance_Difference', ascending=False)
                
                # Save to CSV
                diff_df.to_csv(
                    os.path.join(output_folder, f"importance_diff_cluster_{i}_vs_{j}.csv"),
                    index=False
                )
                
                # Plot top differentiating features
                top_n = 10
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    x='Importance_Difference', 
                    y='Feature', 
                    data=diff_df.head(top_n), 
                    palette='viridis'
                )
                plt.title(f"Top Features Differentiating Cluster {i} vs Cluster {j}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f"importance_diff_cluster_{i}_vs_{j}.png"))
                plt.close()
                
                # Print top 5 differentiating features
                print(f"\nTop differentiating features between Cluster {i} and Cluster {j}:")
                for idx, row in diff_df.head(5).iterrows():
                    print(f"  {row['Feature']}: {row['Importance_Difference']:.4f}")
    
    print(f"\nAll analysis results saved to {output_folder}")
    return output_folder


# Example usage
if __name__ == "__main__":
    # Use your specific folder
    # input_folder = r"RESULTS\2025-03-27_13-51-38"
    # input_folder = r"RESULTS\2025-03-27_17-15-51_data1"
    input_folder = r"RESULTS\2025-03-27_17-12-57_data2"
    analyze_shap_contributions(input_folder)