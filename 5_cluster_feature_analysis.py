import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats

def analyze_cluster_differences(input_folder, output_folder=None, method='sgs'):
    """
    Analyze and visualize the contributions of different features to the differences between clusters.
    
    Parameters:
    input_folder (str): Path to the folder containing the results files
    output_folder (str, optional): Path to save the output files. If None, creates a new folder
    method (str): Method to use for calculating contributions:
        'mean_diff': Based on absolute standardized mean differences
        'ttest': Based on t-test statistics
        'sgs': Scaled Grosse-Stander Analysis (combines magnitude and significance)
    
    Returns:
    str: Path to the output folder
    """
    # Create output folder if not provided
    if output_folder is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = f"Results_SGS/Cluster_Contributions_{timestamp}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created directory: {output_folder}")
    
    # Load the data
    original_data = pd.read_csv(os.path.join(input_folder, "original_data.csv"))
    cluster_labels = pd.read_csv(os.path.join(input_folder, "cluster_labels.csv"))["Cluster Labels"]
    
    # Extract feature columns (drop non-feature columns)
    columns_to_drop = []
    for col in ['Name', 'TARGET', 'Index', 'cmc']:
        if col in original_data.columns:
            columns_to_drop.append(col)
    
    # Get feature data
    feature_data = original_data.copy()
    feature_data.drop(columns=columns_to_drop, axis=1, inplace=True)
    feature_columns = feature_data.columns
    
    # Add cluster labels to the feature data
    feature_data['Cluster'] = cluster_labels
    
    # Get unique cluster IDs
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Create a dictionary to store feature means per cluster
    cluster_means = {}
    for cluster in unique_clusters:
        cluster_means[cluster] = feature_data[feature_data['Cluster'] == cluster][feature_columns].mean()
    
    # Create a dictionary to store feature standard deviations per cluster
    cluster_stds = {}
    for cluster in unique_clusters:
        cluster_stds[cluster] = feature_data[feature_data['Cluster'] == cluster][feature_columns].std()
    
    # Function to calculate contribution scores between two clusters
    def calculate_contributions(cluster1, cluster2, method='sgs'):
        """
        Calculate the contribution of each feature to the difference between two clusters.
        
        Parameters:
        cluster1, cluster2: Cluster IDs to compare
        method: Method to use for calculating contributions
            'mean_diff': Based on absolute standardized mean differences
            'ttest': Based on t-test statistics
            'sgs': Scaled Grosse-Stander Analysis (combines magnitude and significance)
            
        Returns:
        dict: Dictionary of feature contributions (normalized to sum to 1)
        dict: Dictionary of feature direction (1 for higher in cluster1, -1 for lower)
        """
        # Get data for each cluster
        data1 = feature_data[feature_data['Cluster'] == cluster1][feature_columns]
        data2 = feature_data[feature_data['Cluster'] == cluster2][feature_columns]
        
        contributions = {}
        directions = {}  # Store direction of difference (which cluster is higher)
        
        for feature in feature_columns:
            values1 = data1[feature].values
            values2 = data2[feature].values
            
            # Skip features with no variance
            if np.std(values1) == 0 and np.std(values2) == 0:
                contributions[feature] = 0
                directions[feature] = 0
                continue
            
            # Calculate mean difference
            mean1 = np.mean(values1)
            mean2 = np.mean(values2)
            mean_diff = mean1 - mean2
            abs_mean_diff = abs(mean_diff)
            
            # Store direction (which cluster has higher value)
            directions[feature] = 1 if mean_diff > 0 else -1
            
            # Different contribution calculation methods
            if method == 'mean_diff':
                # Just use the absolute mean difference
                contribution = abs_mean_diff
            
            elif method == 'ttest':
                # Use t-test statistic
                t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                contribution = abs(t_stat) if p_value < 0.05 else 0
            
            elif method == 'sgs':
                # Scaled Grosse-Stander Analysis (combines both)
                t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                
                # If not significant or very small difference, set to 0
                if p_value >= 0.05 or abs_mean_diff < 0.001:
                    contribution = 0
                else:
                    # Scale by statistical significance and effect size
                    significance = -np.log10(p_value)
                    contribution = abs_mean_diff * significance
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            contributions[feature] = contribution
        
        # Normalize to sum to 1 (as percentage)
        total = sum(contributions.values())
        if total > 0:
            for feature in contributions:
                contributions[feature] = contributions[feature] / total
        
        return contributions, directions
    
    # Compare all clusters to each other
    for cluster1_idx, cluster1 in enumerate(unique_clusters):
        for cluster2 in unique_clusters[cluster1_idx+1:]:
            # Skip comparison if both are noise
            if cluster1 == -1 and cluster2 == -1:
                continue
            
            # Calculate contributions using selected method
            contributions, directions = calculate_contributions(cluster1, cluster2, method=method)
            
            # Sort features by contribution
            sorted_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            
            # Filter to include only features with significant contributions (> 0.05 or 5%)
            significant_features = [(feature, value) for feature, value in sorted_features if value > 0.05]
            
            # Create bar plot of contributions
            if significant_features:
                features, values = zip(*significant_features)
                
                # Get direction for each feature
                feature_directions = [directions[feature] for feature in features]
                
                plt.figure(figsize=(10, 6))
                
                # Create colorful bars
                # Use different colors for each bar
                colors = plt.cm.tab20(np.linspace(0, 1, len(features)))
                
                bars = plt.bar(features, values, color=colors)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom')
                
                plt.ylabel('% Contribution')
                plt.title(f'Cluster {cluster1} was compared with Cluster {cluster2} using SGS:')
                plt.ylim(0, max(values) * 1.1)  # Add some space at the top for labels
                
                # Add a legend for variables
                plt.legend(['Variables'], loc='upper right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f'Cluster_{cluster1}_vs_Cluster_{cluster2}_SGS.png'))
                plt.close()
                
                # Create a more detailed report with mean values for each feature
                report_df = pd.DataFrame()
                for feature in features:
                    report_df.loc[feature, f'Cluster_{cluster1}_Mean'] = cluster_means[cluster1][feature]
                    report_df.loc[feature, f'Cluster_{cluster2}_Mean'] = cluster_means[cluster2][feature]
                    report_df.loc[feature, f'Cluster_{cluster1}_StdDev'] = cluster_stds[cluster1][feature]
                    report_df.loc[feature, f'Cluster_{cluster2}_StdDev'] = cluster_stds[cluster2][feature]
                    report_df.loc[feature, 'Mean_Difference'] = cluster_means[cluster1][feature] - cluster_means[cluster2][feature]
                    report_df.loc[feature, 'Contribution'] = contributions[feature]
                
                # Sort by contribution
                report_df = report_df.sort_values('Contribution', ascending=False)
                
                # Save the detailed report
                report_df.to_csv(os.path.join(output_folder, f'Cluster_{cluster1}_vs_Cluster_{cluster2}_detailed.csv'))
                
                # Save the simplified contribution data
                contribution_df = pd.DataFrame({
                    'Feature': features,
                    'Contribution': values,
                    'Higher_In_Cluster': [cluster1 if d > 0 else cluster2 for d in feature_directions]
                })
                contribution_df.to_csv(os.path.join(output_folder, f'Cluster_{cluster1}_vs_Cluster_{cluster2}_contributions.csv'), index=False)
                
                # Print the top features
                print(f"Top distinguishing features between Cluster {cluster1} and Cluster {cluster2}:")
                for i, (feature, value) in enumerate(significant_features[:5]):  # Print top 5
                    higher_cluster = cluster1 if directions[feature] > 0 else cluster2
                    print(f"  {feature}: {value:.2f} (higher in Cluster {higher_cluster})")
                print()
    
    # For each cluster, also compare to all other points combined
    for cluster in unique_clusters:
        # Skip noise
        if cluster == -1:
            continue
        
        # Create a temporary binary label: this cluster vs. everything else
        temp_cluster = -999  # Temporary ID for "all other points"
        feature_data['temp_label'] = np.where(feature_data['Cluster'] == cluster, cluster, temp_cluster)
        
        # Calculate contributions
        contributions, directions = calculate_contributions(cluster, temp_cluster, method=method)
        
        # Sort features by contribution
        sorted_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        # Filter to include only features with significant contributions
        significant_features = [(feature, value) for feature, value in sorted_features if value > 0.05]
        
        # Create bar plot of contributions
        if significant_features:
            features, values = zip(*significant_features)
            
            # Get direction for each feature
            feature_directions = [directions[feature] for feature in features]
            
            plt.figure(figsize=(10, 6))
            
            # Create colorful bars
            colors = plt.cm.tab20(np.linspace(0, 1, len(features)))
            
            bars = plt.bar(features, values, color=colors)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.ylabel('% Contribution')
            plt.title(f'Cluster {cluster} was compared with All Others using SGS:')
            plt.ylim(0, max(values) * 1.1)  # Add some space at the top for labels
            
            # Add a legend for feature explanation
            plt.legend(['Variables'], loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'Cluster_{cluster}_vs_All_SGS.png'))
            plt.close()
            
            # Save the contribution data
            contribution_df = pd.DataFrame({
                'Feature': features,
                'Contribution': values,
                'Higher_In_Cluster': ['This cluster' if d > 0 else 'Others' for d in feature_directions]
            })
            contribution_df.to_csv(os.path.join(output_folder, f'Cluster_{cluster}_vs_All_contributions.csv'), index=False)
    
    # Create a summary report with average feature values per cluster
    means_df = pd.DataFrame(cluster_means).T
    means_df.index.name = 'Cluster'
    means_df.to_csv(os.path.join(output_folder, 'cluster_feature_means.csv'))
    
    # Create a heatmap of cluster means for key features
    # Select top contributing features from all comparisons
    all_significant_features = set()
    for cluster1_idx, cluster1 in enumerate(unique_clusters):
        for cluster2 in unique_clusters[cluster1_idx+1:]:
            if cluster1 == -1 and cluster2 == -1:
                continue
                
            contributions, _ = calculate_contributions(cluster1, cluster2, method=method)
            significant_features = [feature for feature, value in contributions.items() if value > 0.05]
            all_significant_features.update(significant_features)
    
    # Convert set to list for pandas DataFrame index
    all_significant_features = list(all_significant_features)  # FIX: Convert set to list
    
    # If we have too many features, limit to top 15
    if len(all_significant_features) > 15:
        # Count feature occurrences across all comparisons
        feature_counts = {}
        for cluster1_idx, cluster1 in enumerate(unique_clusters):
            for cluster2 in unique_clusters[cluster1_idx+1:]:
                if cluster1 == -1 and cluster2 == -1:
                    continue
                
                contributions, _ = calculate_contributions(cluster1, cluster2, method=method)
                for feature, value in contributions.items():
                    if value > 0.05:
                        if feature not in feature_counts:
                            feature_counts[feature] = 0
                        feature_counts[feature] += 1
        
        # Get top 15 most frequently occurring significant features
        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        all_significant_features = [feature for feature, _ in top_features]
    
    # Create a heatmap of cluster means for significant features
    if all_significant_features:
        plt.figure(figsize=(12, 8))
        
        # Create a DataFrame with just the significant features for each cluster
        heatmap_data = pd.DataFrame(index=all_significant_features, columns=unique_clusters)
        
        for cluster in unique_clusters:
            for feature in all_significant_features:
                heatmap_data.loc[feature, cluster] = cluster_means[cluster][feature]
        
        # Scale features for the heatmap (row-wise standardization)
        heatmap_data_scaled = heatmap_data.copy()
        for feature in heatmap_data_scaled.index:
            mean = heatmap_data_scaled.loc[feature].mean()
            std = heatmap_data_scaled.loc[feature].std()
            if std > 0:
                heatmap_data_scaled.loc[feature] = (heatmap_data_scaled.loc[feature] - mean) / std
        
        # Create the heatmap
        plt.imshow(heatmap_data_scaled.values, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Standardized Value')
        
        # Add labels
        plt.yticks(range(len(all_significant_features)), all_significant_features)
        plt.xticks(range(len(unique_clusters)), [f'Cluster {c}' for c in unique_clusters])
        
        plt.title('Feature Values Across Clusters (Standardized)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'feature_cluster_heatmap.png'))
        plt.close()
        
        # Save the raw data for the heatmap
        heatmap_data.to_csv(os.path.join(output_folder, 'feature_cluster_values.csv'))
    
    print(f"All analysis results saved to {output_folder}")
    return output_folder

# Example usage
if __name__ == "__main__":
    # Use your specific folder
    input_folder = r"RESULTS\2025-03-09_20-03-29-logcmc"
    analyze_cluster_differences(input_folder, method='sgs')