import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
import warnings
import argparse
import os
import datetime
import sys
warnings.filterwarnings('ignore')

def parse_arguments():
    """
    Parse command line arguments for file paths
    
    Returns:
    argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Perform SHAP analysis on surfactant data')
    
    parser.add_argument('--descriptors', type=str, 
                        default='molecular_descriptors_concat_data_produc2_Targetcmc.csv',
                        help='Path to the molecular descriptors CSV file')
    
    parser.add_argument('--clusters', type=str, 
                        default='cluster_labels.csv',
                        help='Path to the cluster labels CSV file')
    
    parser.add_argument('--output', type=str, 
                        default='RESULTS',
                        help='Base directory to save output plots (default: RESULTS)')
    
    return parser.parse_args()

def create_output_directory(base_dir):
    """
    Create a timestamped output directory within RESULTS/shap
    
    Parameters:
    base_dir (str): Base directory path
    
    Returns:
    str: Full path to the created output directory
    """
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create path: RESULTS/shap/YYYY-MM-DD_HH-MM-SS
    results_folder = os.path.join(base_dir, 'shap', timestamp)
    
    # Create the directory
    os.makedirs(results_folder, exist_ok=True)
    print(f"Created output directory: {results_folder}")
    
    return results_folder

def check_file_exists(file_path):
    """
    Check if a file exists and print helpful error message if not
    
    Parameters:
    file_path (str): Path to the file to check
    
    Returns:
    bool: True if file exists, False otherwise
    """
    if not os.path.isfile(file_path):
        print(f"ERROR: File not found: {file_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("Please check the file path and try again.")
        return False
    return True

def load_data(descriptors_file, clusters_file):
    """
    Load and merge the molecular descriptors and cluster labels
    
    Parameters:
    descriptors_file (str): Path to the descriptors CSV file
    clusters_file (str): Path to the cluster labels CSV file
    
    Returns:
    pandas.DataFrame: Combined dataframe with features and targets
    """
    # Check if files exist
    if not check_file_exists(descriptors_file) or not check_file_exists(clusters_file):
        sys.exit(1)
    
    try:
        # Load data
        descriptors_df = pd.read_csv(descriptors_file)
        clusters_df = pd.read_csv(clusters_file)
        
        # Ensure both datasets have same number of rows
        if len(descriptors_df) != len(clusters_df):
            print(f"ERROR: Dataset lengths don't match! Descriptors: {len(descriptors_df)}, Clusters: {len(clusters_df)}")
            sys.exit(1)
        
        # Add cluster labels to the descriptors dataframe
        descriptors_df['Cluster'] = clusters_df['Cluster Labels']
        
        print(f"Loaded {len(descriptors_df)} samples with {len(descriptors_df.columns)} columns")
        
        return descriptors_df
    
    except Exception as e:
        print(f"ERROR loading data: {str(e)}")
        sys.exit(1)

def prepare_features(df):
    """
    Prepare features for analysis by removing non-feature columns
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    tuple: X (features), y_cmc (CMC values), y_cluster (cluster labels)
    """
    # Identify feature columns (numerical columns except targets)
    non_feature_cols = ['Name', 'SMILES', 'TARGET', 'Log(cmc*1000+1)', 'Cluster']
    
    # Make sure to only exclude columns that actually exist
    non_feature_cols = [col for col in non_feature_cols if col in df.columns]
    
    feature_cols = [col for col in df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Extract features and targets
    X = df[feature_cols]
    
    # Check if target columns exist
    if 'TARGET' not in df.columns:
        print("WARNING: 'TARGET' column not found in descriptors data. Using dummy values.")
        y_cmc = pd.Series(np.zeros(len(df)))
    else:
        y_cmc = df['TARGET']
    
    if 'Cluster' not in df.columns:
        print("WARNING: 'Cluster' column not found. Using dummy values.")
        y_cluster = pd.Series(np.zeros(len(df), dtype=int))
    else:
        y_cluster = df['Cluster']
    
    print(f"Selected {len(feature_cols)} features for analysis")
    print(f"CMC target range: {y_cmc.min()} to {y_cmc.max()}")
    print(f"Cluster distribution: {y_cluster.value_counts().to_dict()}")
    
    return X, y_cmc, y_cluster, feature_cols

def train_models(X, y_cmc, y_cluster):
    """
    Train RandomForest models for both regression (CMC values) and classification (clusters)
    
    Parameters:
    X (pandas.DataFrame): Feature matrix
    y_cmc (pandas.Series): CMC target values
    y_cluster (pandas.Series): Cluster labels
    
    Returns:
    tuple: Trained regression model, trained classification model
    """
    try:
        # Split data
        X_train, X_test, y_cmc_train, y_cmc_test, y_cluster_train, y_cluster_test = train_test_split(
            X, y_cmc, y_cluster, test_size=0.25, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train regression model for CMC values
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train_scaled, y_cmc_train)
        cmc_score = rf_regressor.score(X_test_scaled, y_cmc_test)
        print(f"CMC regression model R² score: {cmc_score:.4f}")
        
        # Train classification model for clusters
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_scaled, y_cluster_train)
        cluster_score = rf_classifier.score(X_test_scaled, y_cluster_test)
        print(f"Cluster classification model accuracy: {cluster_score:.4f}")
        
        return rf_regressor, rf_classifier, scaler, X_train, X_test
    
    except Exception as e:
        print(f"ERROR training models: {str(e)}")
        sys.exit(1)

def calculate_feature_importance(model, X, feature_names, plot_file, title, output_dir, is_classifier=False):
    """
    Calculate and plot feature importance for a model using built-in importance
    
    Parameters:
    model: Trained model
    X: Feature data
    feature_names: List of feature names
    plot_file: Output file name
    title: Plot title
    output_dir: Output directory
    is_classifier: Whether the model is a classifier
    """
    try:
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(title, fontsize=16)
        plt.bar(range(len(indices)), importances[indices], color='b', align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, min(20, len(indices))])  # Show top 20 features at most
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, plot_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {plot_path}")
        
        # Return top feature indices
        return indices
        
    except Exception as e:
        print(f"ERROR calculating feature importance: {str(e)}")
        return np.argsort(model.feature_importances_)[::-1] if hasattr(model, 'feature_importances_') else []

def plot_feature_distributions_by_cluster(df, feature_names, top_features, output_dir):
    """
    Plot the distributions of top features across clusters
    
    Parameters:
    df: Full dataframe with cluster labels
    feature_names: List of feature names
    top_features: List of top feature indices
    output_dir: Output directory
    """
    try:
        num_features = min(5, len(top_features))
        selected_features = [feature_names[i] for i in top_features[:num_features]]
        
        # Create a figure with subplots
        fig, axes = plt.subplots(num_features, 1, figsize=(12, 4 * num_features))
        if num_features == 1:
            axes = [axes]
        
        # Plot each feature
        for i, feature in enumerate(selected_features):
            sns.boxplot(x='Cluster', y=feature, data=df, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature} by Cluster', fontsize=12)
            axes[i].set_xlabel('Cluster', fontsize=10)
            axes[i].set_ylabel(feature, fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "feature_distributions_by_cluster.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {plot_path}")
        
    except Exception as e:
        print(f"ERROR plotting feature distributions: {str(e)}")

def plot_partial_dependence(model, X, feature_names, output_dir, is_classifier=False):
    """
    Create partial dependence plots for top features
    
    Parameters:
    model: Trained model
    X: Feature data
    feature_names: List of feature names
    output_dir: Output directory
    is_classifier: Whether the model is a classifier
    """
    try:
        # Get top feature importances
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:5]  # Top 5 features
        
        from sklearn.inspection import partial_dependence, plot_partial_dependence
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Title for the overall figure
        fig.suptitle("Partial Dependence of Top Features", fontsize=16)
        
        # For each top feature
        for i, idx in enumerate(top_indices):
            if i >= len(axes) - 1:  # Skip the last subplot
                break
                
            feature_name = feature_names[idx]
            
            # Calculate partial dependence
            pdp = partial_dependence(model, X, [idx], kind="average")
            feature_values = pdp["values"][0]
            pdp_values = pdp["average"][0]
            
            # Plot
            axes[i].plot(feature_values, pdp_values)
            axes[i].set_xlabel(feature_name)
            axes[i].set_ylabel("Partial Dependence")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        
        # Remove any unused subplots
        for i in range(len(top_indices), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save plot
        plot_path = os.path.join(output_dir, "partial_dependence.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {plot_path}")
        
    except Exception as e:
        print(f"ERROR creating partial dependence plots: {str(e)}")

def save_analysis_info(output_dir, X, y_cmc, y_cluster, feature_names, 
                       rf_regressor, rf_classifier, descriptors_file, clusters_file):
    """
    Save analysis information to a text file
    
    Parameters:
    output_dir: Output directory path
    X: Feature matrix
    y_cmc: CMC target values
    y_cluster: Cluster labels
    feature_names: List of feature names
    rf_regressor: Trained regression model
    rf_classifier: Trained classification model
    descriptors_file: Path to descriptors file
    clusters_file: Path to clusters file
    """
    try:
        info_file = os.path.join(output_dir, "analysis_info.txt")
        
        with open(info_file, 'w') as f:
            f.write("SHAP Analysis for Surfactant Properties\n")
            f.write("=====================================\n\n")
            
            f.write(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Input Files:\n")
            f.write(f"  Descriptors: {descriptors_file}\n")
            f.write(f"  Clusters: {clusters_file}\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  Number of samples: {len(X)}\n")
            f.write(f"  Number of features: {len(feature_names)}\n")
            f.write(f"  CMC value range: {y_cmc.min():.4f} to {y_cmc.max():.4f}\n")
            f.write(f"  Number of clusters: {len(np.unique(y_cluster))}\n")
            f.write(f"  Cluster distribution: {y_cluster.value_counts().to_dict()}\n\n")
            
            f.write("Model Performance:\n")
            f.write(f"  CMC Regression R² score: {rf_regressor.score(X, y_cmc):.4f}\n")
            f.write(f"  Cluster Classification accuracy: {rf_classifier.score(X, y_cluster):.4f}\n\n")
            
            # Top 10 features for CMC prediction
            reg_importance = rf_regressor.feature_importances_
            top_cmc_idx = np.argsort(reg_importance)[::-1][:10]
            
            f.write("Top 10 Features for CMC Prediction:\n")
            for i, idx in enumerate(top_cmc_idx):
                f.write(f"  {i+1}. {feature_names[idx]}: {reg_importance[idx]:.4f}\n")
            f.write("\n")
            
            # Top 10 features for cluster classification
            cls_importance = rf_classifier.feature_importances_
            top_cluster_idx = np.argsort(cls_importance)[::-1][:10]
            
            f.write("Top 10 Features for Cluster Classification:\n")
            for i, idx in enumerate(top_cluster_idx):
                f.write(f"  {i+1}. {feature_names[idx]}: {cls_importance[idx]:.4f}\n")
        
        print(f"Saved analysis information to {info_file}")
    
    except Exception as e:
        print(f"ERROR saving analysis info: {str(e)}")

def create_correlation_heatmap(X, feature_names, output_dir):
    """
    Create a heatmap of feature correlations
    
    Parameters:
    X: Feature matrix
    feature_names: List of feature names
    output_dir: Output directory
    """
    try:
        # Calculate correlation matrix
        corr_matrix = X.corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
                    center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "feature_correlation_heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {plot_path}")
        
    except Exception as e:
        print(f"ERROR creating correlation heatmap: {str(e)}")

def main():
    """
    Main function to run the SHAP analysis
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # File paths from arguments
        descriptors_file = args.descriptors
        clusters_file = args.clusters
        base_output_dir = args.output
        
        # Create timestamped output directory
        output_dir = create_output_directory(base_output_dir)
        
        print(f"Using descriptors file: {descriptors_file}")
        print(f"Using clusters file: {clusters_file}")
        
        # Load and prepare data
        df = load_data(descriptors_file, clusters_file)
        X, y_cmc, y_cluster, feature_names = prepare_features(df)
        
        # Create correlation heatmap
        create_correlation_heatmap(X, feature_names, output_dir)
        
        # Train models
        rf_regressor, rf_classifier, scaler, X_train, X_test = train_models(X, y_cmc, y_cluster)
        
        # Calculate and plot feature importance using built-in methods instead of SHAP
        print("\nCalculating feature importance for CMC prediction...")
        top_cmc_features = calculate_feature_importance(
            rf_regressor, X, feature_names, 
            "cmc_feature_importance.png", 
            "Feature Importance for CMC Prediction", 
            output_dir
        )
        
        print("\nCalculating feature importance for cluster classification...")
        top_cluster_features = calculate_feature_importance(
            rf_classifier, X, feature_names, 
            "cluster_feature_importance.png", 
            "Feature Importance for Cluster Classification", 
            output_dir, 
            is_classifier=True
        )
        
        # Partial dependence plots
        print("\nCreating partial dependence plots...")
        plot_partial_dependence(rf_regressor, X, feature_names, output_dir)
        
        # Plot feature distributions by cluster
        print("\nPlotting feature distributions by cluster...")
        plot_feature_distributions_by_cluster(df, feature_names, top_cluster_features, output_dir)
        
        # Print top features for CMC prediction
        print("\nTop 10 features for CMC prediction:")
        for i in range(min(10, len(feature_names))):
            idx = top_cmc_features[i]
            print(f"{feature_names[idx]}: {rf_regressor.feature_importances_[idx]:.4f}")
        
        # Print top features for cluster classification
        print("\nTop 10 features for cluster classification:")
        for i in range(min(10, len(feature_names))):
            idx = top_cluster_features[i]
            print(f"{feature_names[idx]}: {rf_classifier.feature_importances_[idx]:.4f}")
        
        # Save analysis information
        save_analysis_info(
            output_dir, X, y_cmc, y_cluster, feature_names, 
            rf_regressor, rf_classifier, descriptors_file, clusters_file
        )
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

#========================================================

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import shap
# import warnings
# import argparse
# import os
# import datetime
# warnings.filterwarnings('ignore')

# def parse_arguments():
#     """
#     Parse command line arguments for file paths
    
#     Returns:
#     argparse.Namespace: Parsed arguments
#     """
#     parser = argparse.ArgumentParser(description='Perform SHAP analysis on surfactant data')
    
#     parser.add_argument('--descriptors', type=str, 
#                         default='./DATA/molecular_descriptors_concat_data_produc2_Targetcmc.csv',
#                         help='Path to the molecular descriptors CSV file')
    
#     parser.add_argument('--clusters', type=str, 
#                         default='./RESULTS/2025-03-06_12-55-55-cmc/cluster_labels.csv',
#                         help='Path to the cluster labels CSV file')
    
#     parser.add_argument('--output', type=str, 
#                         default='./RESULTS',
#                         help='Base directory to save output plots (default: RESULTS)')
    
#     return parser.parse_args()

# def create_output_directory(base_dir):
#     """
#     Create a timestamped output directory within RESULTS/shap
    
#     Parameters:
#     base_dir (str): Base directory path
    
#     Returns:
#     str: Full path to the created output directory
#     """
#     # Create timestamp
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
#     # Create path: RESULTS/shap/YYYY-MM-DD_HH-MM-SS
#     results_folder = os.path.join(base_dir, 'shap', timestamp)
    
#     # Create the directory
#     os.makedirs(results_folder, exist_ok=True)
#     print(f"Created output directory: {results_folder}")
    
#     return results_folder

# def check_file_exists(file_path):
#     """
#     Check if a file exists and print helpful error message if not
    
#     Parameters:
#     file_path (str): Path to the file to check
    
#     Returns:
#     bool: True if file exists, False otherwise
#     """
#     if not os.path.isfile(file_path):
#         print(f"ERROR: File not found: {file_path}")
#         print(f"Current working directory: {os.getcwd()}")
#         print("Please check the file path and try again.")
#         return False
#     return True

# def load_data(descriptors_file, clusters_file):
#     """
#     Load and merge the molecular descriptors and cluster labels
    
#     Parameters:
#     descriptors_file (str): Path to the descriptors CSV file
#     clusters_file (str): Path to the cluster labels CSV file
    
#     Returns:
#     pandas.DataFrame: Combined dataframe with features and targets
#     """
#     # Load data
#     descriptors_df = pd.read_csv(descriptors_file)
#     clusters_df = pd.read_csv(clusters_file)
    
#     # Ensure both datasets have same number of rows
#     assert len(descriptors_df) == len(clusters_df), "Dataset lengths don't match"
    
#     # Add cluster labels to the descriptors dataframe
#     descriptors_df['Cluster'] = clusters_df['Cluster Labels']
    
#     print(f"Loaded {len(descriptors_df)} samples with {len(descriptors_df.columns)} columns")
    
#     return descriptors_df

# def prepare_features(df):
#     """
#     Prepare features for analysis by removing non-feature columns
    
#     Parameters:
#     df (pandas.DataFrame): Input dataframe
    
#     Returns:
#     tuple: X (features), y_cmc (CMC values), y_cluster (cluster labels)
#     """
#     # Identify feature columns (numerical columns except targets)
#     non_feature_cols = ['Name', 'SMILES', 'TARGET', 'Log(cmc*1000+1)', 'Cluster']
#     feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
#     # Extract features and targets
#     X = df[feature_cols]
#     y_cmc = df['TARGET']
#     y_cluster = df['Cluster']
    
#     print(f"Selected {len(feature_cols)} features for analysis")
#     print(f"CMC target range: {y_cmc.min()} to {y_cmc.max()}")
#     print(f"Cluster distribution: {y_cluster.value_counts().to_dict()}")
    
#     return X, y_cmc, y_cluster, feature_cols

# def train_models(X, y_cmc, y_cluster):
#     """
#     Train RandomForest models for both regression (CMC values) and classification (clusters)
    
#     Parameters:
#     X (pandas.DataFrame): Feature matrix
#     y_cmc (pandas.Series): CMC target values
#     y_cluster (pandas.Series): Cluster labels
    
#     Returns:
#     tuple: Trained regression model, trained classification model
#     """
#     # Split data
#     X_train, X_test, y_cmc_train, y_cmc_test, y_cluster_train, y_cluster_test = train_test_split(
#         X, y_cmc, y_cluster, test_size=0.25, random_state=42
#     )
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Train regression model for CMC values
#     rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_regressor.fit(X_train_scaled, y_cmc_train)
#     cmc_score = rf_regressor.score(X_test_scaled, y_cmc_test)
#     print(f"CMC regression model R² score: {cmc_score:.4f}")
    
#     # Train classification model for clusters
#     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf_classifier.fit(X_train_scaled, y_cluster_train)
#     cluster_score = rf_classifier.score(X_test_scaled, y_cluster_test)
#     print(f"Cluster classification model accuracy: {cluster_score:.4f}")
    
#     return rf_regressor, rf_classifier, scaler, X_train, X_test

# def calculate_shap_values(model, X_train, feature_names, is_classifier=False):
#     """
#     Calculate SHAP values for the trained model
    
#     Parameters:
#     model: Trained model
#     X_train: Training data
#     feature_names: List of feature names
#     is_classifier: Whether the model is a classifier
    
#     Returns:
#     shap.Explainer: SHAP explainer object
#     """
#     # Initialize SHAP explainer
#     explainer = shap.TreeExplainer(model)
    
#     # Calculate SHAP values on training data (using a subset for efficiency if needed)
#     sample_size = min(100, len(X_train))
#     X_sample = X_train.sample(sample_size, random_state=42)
    
#     # Calculate SHAP values
#     shap_values = explainer.shap_values(X_sample)
    
#     # For classifiers, shap_values is a list of arrays (one per class)
#     # For regressors, shap_values is a single array
#     return explainer, shap_values, X_sample

# def plot_shap_summary(shap_values, X_sample, feature_names, title, class_index=None):
#     """
#     Plot SHAP summary plot
    
#     Parameters:
#     shap_values: SHAP values
#     X_sample: Feature values
#     feature_names: List of feature names
#     title: Plot title
#     class_index: Index of the class to plot (for classifiers)
#     """
#     plt.figure(figsize=(12, 8))
    
#     if class_index is not None:
#         # For classifiers, we need to specify which class to plot
#         shap.summary_plot(
#             shap_values[class_index], 
#             X_sample, 
#             feature_names=feature_names,
#             show=False
#         )
#     else:
#         # For regressors, we can just plot the single set of SHAP values
#         shap.summary_plot(
#             shap_values, 
#             X_sample, 
#             feature_names=feature_names,
#             show=False
#         )
    
#     plt.title(title, fontsize=14)
#     plt.tight_layout()
#     return plt.gcf()

# def create_feature_importance_plots(regressor, classifier, X_train, X_test, feature_names):
#     """
#     Create feature importance plots using SHAP values
    
#     Parameters:
#     regressor: Trained regression model
#     classifier: Trained classification model
#     X_train: Training data
#     X_test: Test data
#     feature_names: List of feature names
    
#     Returns:
#     dict: Dictionary of plot figures
#     """
#     plots = {}
    
#     # CMC regression model SHAP analysis
#     print("Calculating SHAP values for CMC regression model...")
#     reg_explainer, reg_shap_values, reg_X_sample = calculate_shap_values(
#         regressor, X_train, feature_names, is_classifier=False
#     )
    
#     # Create summary plot for regression model
#     plt.figure(figsize=(12, 10))
#     shap.summary_plot(
#         reg_shap_values, 
#         reg_X_sample,
#         feature_names=feature_names,
#         show=False
#     )
#     plt.title("SHAP Feature Importance for CMC Values", fontsize=14)
#     plt.tight_layout()
#     plots['cmc_summary'] = plt.gcf()
    
#     # Create bar plot for regression model
#     plt.figure(figsize=(12, 8))
#     shap.summary_plot(
#         reg_shap_values, 
#         reg_X_sample,
#         feature_names=feature_names,
#         plot_type="bar",
#         show=False
#     )
#     plt.title("SHAP Mean Absolute Impact on CMC Values", fontsize=14)
#     plt.tight_layout()
#     plots['cmc_bar'] = plt.gcf()
    
#     # Cluster classification model SHAP analysis
#     print("Calculating SHAP values for cluster classification model...")
#     cls_explainer, cls_shap_values, cls_X_sample = calculate_shap_values(
#         classifier, X_train, feature_names, is_classifier=True
#     )
    
#     # Create summary plot for classification model (using mean absolute SHAP values across classes)
#     # Convert list of arrays to a single array by taking the mean absolute value
#     mean_abs_shap = np.abs(np.array(cls_shap_values)).mean(axis=0)
    
#     plt.figure(figsize=(12, 8))
#     shap.summary_plot(
#         mean_abs_shap, 
#         cls_X_sample,
#         feature_names=feature_names,
#         plot_type="bar",
#         show=False
#     )
#     plt.title("SHAP Feature Importance for Cluster Classification", fontsize=14)
#     plt.tight_layout()
#     plots['cluster_bar'] = plt.gcf()
    
#     # Create individual plots for each cluster
#     unique_clusters = len(cls_shap_values)
#     for i in range(unique_clusters):
#         plt.figure(figsize=(12, 8))
#         shap.summary_plot(
#             cls_shap_values[i], 
#             cls_X_sample,
#             feature_names=feature_names,
#             show=False
#         )
#         plt.title(f"SHAP Values for Cluster {i}", fontsize=14)
#         plt.tight_layout()
#         plots[f'cluster_{i}'] = plt.gcf()
    
#     return plots

# def plot_feature_distributions_by_cluster(df, X, top_features, feature_names):
#     """
#     Plot the distributions of top features across clusters
    
#     Parameters:
#     df: Full dataframe with cluster labels
#     X: Feature matrix
#     top_features: List of top feature indices
#     feature_names: List of feature names
    
#     Returns:
#     matplotlib.figure.Figure: Figure object
#     """
#     num_features = min(5, len(top_features))
#     selected_features = [feature_names[i] for i in top_features[:num_features]]
    
#     # Create a figure with subplots
#     fig, axes = plt.subplots(num_features, 1, figsize=(12, 4 * num_features))
#     if num_features == 1:
#         axes = [axes]
    
#     # Plot each feature
#     for i, feature in enumerate(selected_features):
#         sns.boxplot(x='Cluster', y=feature, data=df, ax=axes[i])
#         axes[i].set_title(f'Distribution of {feature} by Cluster', fontsize=12)
#         axes[i].set_xlabel('Cluster', fontsize=10)
#         axes[i].set_ylabel(feature, fontsize=10)
    
#     plt.tight_layout()
#     return fig

# def save_analysis_info(output_dir, X, y_cmc, y_cluster, feature_names, 
#                        rf_regressor, rf_classifier, descriptors_file, clusters_file):
#     """
#     Save analysis information to a text file
    
#     Parameters:
#     output_dir: Output directory path
#     X: Feature matrix
#     y_cmc: CMC target values
#     y_cluster: Cluster labels
#     feature_names: List of feature names
#     rf_regressor: Trained regression model
#     rf_classifier: Trained classification model
#     descriptors_file: Path to descriptors file
#     clusters_file: Path to clusters file
#     """
#     info_file = os.path.join(output_dir, "analysis_info.txt")
    
#     with open(info_file, 'w') as f:
#         f.write("SHAP Analysis for Surfactant Properties\n")
#         f.write("=====================================\n\n")
        
#         f.write(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
#         f.write("Input Files:\n")
#         f.write(f"  Descriptors: {descriptors_file}\n")
#         f.write(f"  Clusters: {clusters_file}\n\n")
        
#         f.write("Dataset Information:\n")
#         f.write(f"  Number of samples: {len(X)}\n")
#         f.write(f"  Number of features: {len(feature_names)}\n")
#         f.write(f"  CMC value range: {y_cmc.min():.4f} to {y_cmc.max():.4f}\n")
#         f.write(f"  Number of clusters: {len(np.unique(y_cluster))}\n")
#         f.write(f"  Cluster distribution: {y_cluster.value_counts().to_dict()}\n\n")
        
#         f.write("Model Performance:\n")
#         f.write(f"  CMC Regression R² score: {rf_regressor.score(X, y_cmc):.4f}\n")
#         f.write(f"  Cluster Classification accuracy: {rf_classifier.score(X, y_cluster):.4f}\n\n")
        
#         # Top 10 features for CMC prediction
#         reg_importance = rf_regressor.feature_importances_
#         top_cmc_idx = np.argsort(reg_importance)[::-1][:10]
        
#         f.write("Top 10 Features for CMC Prediction:\n")
#         for i, idx in enumerate(top_cmc_idx):
#             f.write(f"  {i+1}. {feature_names[idx]}: {reg_importance[idx]:.4f}\n")
#         f.write("\n")
        
#         # Top 10 features for cluster classification
#         cls_importance = rf_classifier.feature_importances_
#         top_cluster_idx = np.argsort(cls_importance)[::-1][:10]
        
#         f.write("Top 10 Features for Cluster Classification:\n")
#         for i, idx in enumerate(top_cluster_idx):
#             f.write(f"  {i+1}. {feature_names[idx]}: {cls_importance[idx]:.4f}\n")
    
#     print(f"Saved analysis information to {info_file}")

# def main():
#     """
#     Main function to run the SHAP analysis
#     """
#     # Parse command line arguments
#     args = parse_arguments()
    
#     # File paths from arguments
#     descriptors_file = args.descriptors
#     clusters_file = args.clusters
#     base_output_dir = args.output
    
#     # Create timestamped output directory
#     output_dir = create_output_directory(base_output_dir)
    
#     print(f"Using descriptors file: {descriptors_file}")
#     print(f"Using clusters file: {clusters_file}")
    
#     # Load and prepare data
#     df = load_data(descriptors_file, clusters_file)
#     X, y_cmc, y_cluster, feature_names = prepare_features(df)
    
#     # Train models
#     rf_regressor, rf_classifier, scaler, X_train, X_test = train_models(X, y_cmc, y_cluster)
    
#     # Create SHAP visualizations
#     plots = create_feature_importance_plots(
#         rf_regressor, rf_classifier, X_train, X_test, feature_names
#     )
    
#     # Get top features for CMC prediction
#     reg_feature_importances = rf_regressor.feature_importances_
#     top_cmc_features = np.argsort(reg_feature_importances)[::-1]
#     print("\nTop 10 features for CMC prediction:")
#     for i in range(10):
#         idx = top_cmc_features[i]
#         print(f"{feature_names[idx]}: {reg_feature_importances[idx]:.4f}")
    
#     # Get top features for cluster classification
#     cls_feature_importances = rf_classifier.feature_importances_
#     top_cluster_features = np.argsort(cls_feature_importances)[::-1]
#     print("\nTop 10 features for cluster classification:")
#     for i in range(10):
#         idx = top_cluster_features[i]
#         print(f"{feature_names[idx]}: {cls_feature_importances[idx]:.4f}")
    
#     # Plot feature distributions by cluster
#     cluster_dist_plot = plot_feature_distributions_by_cluster(
#         df, X, top_cluster_features, feature_names
#     )
    
#     # Save plots
#     for name, plot in plots.items():
#         filename = os.path.join(output_dir, f"shap_{name}.png")
#         plot.savefig(filename, dpi=300, bbox_inches='tight')
#         print(f"Saved {filename}")
    
#     cluster_dist_filename = os.path.join(output_dir, "feature_distributions_by_cluster.png")
#     cluster_dist_plot.savefig(cluster_dist_filename, dpi=300, bbox_inches='tight')
#     print(f"Saved {cluster_dist_filename}")
    
#     # Save analysis information
#     save_analysis_info(
#         output_dir, X, y_cmc, y_cluster, feature_names, 
#         rf_regressor, rf_classifier, descriptors_file, clusters_file
#     )
    
#     # Show plots if in interactive mode
#     if plt.isinteractive():
#         plt.show()
#     else:
#         plt.close('all')
        
#     print(f"\nAnalysis complete! Results saved to: {output_dir}")

# if __name__ == "__main__":
#     main()

#---------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import shap
# import warnings
# warnings.filterwarnings('ignore')

# def load_data(descriptors_file, clusters_file):
#     """
#     Load and merge the molecular descriptors and cluster labels
    
#     Parameters:
#     descriptors_file (str): Path to the descriptors CSV file
#     clusters_file (str): Path to the cluster labels CSV file
    
#     Returns:
#     pandas.DataFrame: Combined dataframe with features and targets
#     """
#     # Load data
#     descriptors_df = pd.read_csv(descriptors_file)
#     clusters_df = pd.read_csv(clusters_file)
    
#     # Ensure both datasets have same number of rows
#     assert len(descriptors_df) == len(clusters_df), "Dataset lengths don't match"
    
#     # Add cluster labels to the descriptors dataframe
#     descriptors_df['Cluster'] = clusters_df['Cluster Labels']
    
#     print(f"Loaded {len(descriptors_df)} samples with {len(descriptors_df.columns)} columns")
    
#     return descriptors_df

# def prepare_features(df):
#     """
#     Prepare features for analysis by removing non-feature columns
    
#     Parameters:
#     df (pandas.DataFrame): Input dataframe
    
#     Returns:
#     tuple: X (features), y_cmc (CMC values), y_cluster (cluster labels)
#     """
#     # Identify feature columns (numerical columns except targets)
#     non_feature_cols = ['Name', 'SMILES', 'TARGET', 'Log(cmc*1000+1)', 'Cluster']
#     feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
#     # Extract features and targets
#     X = df[feature_cols]
#     y_cmc = df['TARGET']
#     y_cluster = df['Cluster']
    
#     print(f"Selected {len(feature_cols)} features for analysis")
#     print(f"CMC target range: {y_cmc.min()} to {y_cmc.max()}")
#     print(f"Cluster distribution: {y_cluster.value_counts().to_dict()}")
    
#     return X, y_cmc, y_cluster, feature_cols

# def train_models(X, y_cmc, y_cluster):
#     """
#     Train RandomForest models for both regression (CMC values) and classification (clusters)
    
#     Parameters:
#     X (pandas.DataFrame): Feature matrix
#     y_cmc (pandas.Series): CMC target values
#     y_cluster (pandas.Series): Cluster labels
    
#     Returns:
#     tuple: Trained regression model, trained classification model
#     """
#     # Split data
#     X_train, X_test, y_cmc_train, y_cmc_test, y_cluster_train, y_cluster_test = train_test_split(
#         X, y_cmc, y_cluster, test_size=0.25, random_state=42
#     )
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Train regression model for CMC values
#     rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_regressor.fit(X_train_scaled, y_cmc_train)
#     cmc_score = rf_regressor.score(X_test_scaled, y_cmc_test)
#     print(f"CMC regression model R² score: {cmc_score:.4f}")
    
#     # Train classification model for clusters
#     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf_classifier.fit(X_train_scaled, y_cluster_train)
#     cluster_score = rf_classifier.score(X_test_scaled, y_cluster_test)
#     print(f"Cluster classification model accuracy: {cluster_score:.4f}")
    
#     return rf_regressor, rf_classifier, scaler, X_train, X_test

# def calculate_shap_values(model, X_train, feature_names, is_classifier=False):
#     """
#     Calculate SHAP values for the trained model
    
#     Parameters:
#     model: Trained model
#     X_train: Training data
#     feature_names: List of feature names
#     is_classifier: Whether the model is a classifier
    
#     Returns:
#     shap.Explainer: SHAP explainer object
#     """
#     # Initialize SHAP explainer
#     explainer = shap.TreeExplainer(model)
    
#     # Calculate SHAP values on training data (using a subset for efficiency if needed)
#     sample_size = min(100, len(X_train))
#     X_sample = X_train.sample(sample_size, random_state=42)
    
#     # Calculate SHAP values
#     shap_values = explainer.shap_values(X_sample)
    
#     # For classifiers, shap_values is a list of arrays (one per class)
#     # For regressors, shap_values is a single array
#     return explainer, shap_values, X_sample

# def plot_shap_summary(shap_values, X_sample, feature_names, title, class_index=None):
#     """
#     Plot SHAP summary plot
    
#     Parameters:
#     shap_values: SHAP values
#     X_sample: Feature values
#     feature_names: List of feature names
#     title: Plot title
#     class_index: Index of the class to plot (for classifiers)
#     """
#     plt.figure(figsize=(12, 8))
    
#     if class_index is not None:
#         # For classifiers, we need to specify which class to plot
#         shap.summary_plot(
#             shap_values[class_index], 
#             X_sample, 
#             feature_names=feature_names,
#             show=False
#         )
#     else:
#         # For regressors, we can just plot the single set of SHAP values
#         shap.summary_plot(
#             shap_values, 
#             X_sample, 
#             feature_names=feature_names,
#             show=False
#         )
    
#     plt.title(title, fontsize=14)
#     plt.tight_layout()
#     return plt.gcf()

# def create_feature_importance_plots(regressor, classifier, X_train, X_test, feature_names):
#     """
#     Create feature importance plots using SHAP values
    
#     Parameters:
#     regressor: Trained regression model
#     classifier: Trained classification model
#     X_train: Training data
#     X_test: Test data
#     feature_names: List of feature names
    
#     Returns:
#     dict: Dictionary of plot figures
#     """
#     plots = {}
    
#     # CMC regression model SHAP analysis
#     print("Calculating SHAP values for CMC regression model...")
#     reg_explainer, reg_shap_values, reg_X_sample = calculate_shap_values(
#         regressor, X_train, feature_names, is_classifier=False
#     )
    
#     # Create summary plot for regression model
#     plt.figure(figsize=(12, 10))
#     shap.summary_plot(
#         reg_shap_values, 
#         reg_X_sample,
#         feature_names=feature_names,
#         show=False
#     )
#     plt.title("SHAP Feature Importance for CMC Values", fontsize=14)
#     plt.tight_layout()
#     plots['cmc_summary'] = plt.gcf()
    
#     # Create bar plot for regression model
#     plt.figure(figsize=(12, 8))
#     shap.summary_plot(
#         reg_shap_values, 
#         reg_X_sample,
#         feature_names=feature_names,
#         plot_type="bar",
#         show=False
#     )
#     plt.title("SHAP Mean Absolute Impact on CMC Values", fontsize=14)
#     plt.tight_layout()
#     plots['cmc_bar'] = plt.gcf()
    
#     # Cluster classification model SHAP analysis
#     print("Calculating SHAP values for cluster classification model...")
#     cls_explainer, cls_shap_values, cls_X_sample = calculate_shap_values(
#         classifier, X_train, feature_names, is_classifier=True
#     )
    
#     # Create summary plot for classification model (using mean absolute SHAP values across classes)
#     # Convert list of arrays to a single array by taking the mean absolute value
#     mean_abs_shap = np.abs(np.array(cls_shap_values)).mean(axis=0)
    
#     plt.figure(figsize=(12, 8))
#     shap.summary_plot(
#         mean_abs_shap, 
#         cls_X_sample,
#         feature_names=feature_names,
#         plot_type="bar",
#         show=False
#     )
#     plt.title("SHAP Feature Importance for Cluster Classification", fontsize=14)
#     plt.tight_layout()
#     plots['cluster_bar'] = plt.gcf()
    
#     # Create individual plots for each cluster
#     unique_clusters = len(cls_shap_values)
#     for i in range(unique_clusters):
#         plt.figure(figsize=(12, 8))
#         shap.summary_plot(
#             cls_shap_values[i], 
#             cls_X_sample,
#             feature_names=feature_names,
#             show=False
#         )
#         plt.title(f"SHAP Values for Cluster {i}", fontsize=14)
#         plt.tight_layout()
#         plots[f'cluster_{i}'] = plt.gcf()
    
#     return plots

# def plot_feature_distributions_by_cluster(df, X, top_features, feature_names):
#     """
#     Plot the distributions of top features across clusters
    
#     Parameters:
#     df: Full dataframe with cluster labels
#     X: Feature matrix
#     top_features: List of top feature indices
#     feature_names: List of feature names
    
#     Returns:
#     matplotlib.figure.Figure: Figure object
#     """
#     num_features = min(5, len(top_features))
#     selected_features = [feature_names[i] for i in top_features[:num_features]]
    
#     # Create a figure with subplots
#     fig, axes = plt.subplots(num_features, 1, figsize=(12, 4 * num_features))
#     if num_features == 1:
#         axes = [axes]
    
#     # Plot each feature
#     for i, feature in enumerate(selected_features):
#         sns.boxplot(x='Cluster', y=feature, data=df, ax=axes[i])
#         axes[i].set_title(f'Distribution of {feature} by Cluster', fontsize=12)
#         axes[i].set_xlabel('Cluster', fontsize=10)
#         axes[i].set_ylabel(feature, fontsize=10)
    
#     plt.tight_layout()
#     return fig

# def main():
#     """
#     Main function to run the SHAP analysis
#     """
#     # File paths
#     descriptors_file = 'molecular_descriptors_concat_data_produc2_Targetcmc.csv'
#     clusters_file = 'cluster_labels.csv'
    
#     # Load and prepare data
#     df = load_data(descriptors_file, clusters_file)
#     X, y_cmc, y_cluster, feature_names = prepare_features(df)
    
#     # Train models
#     rf_regressor, rf_classifier, scaler, X_train, X_test = train_models(X, y_cmc, y_cluster)
    
#     # Create SHAP visualizations
#     plots = create_feature_importance_plots(
#         rf_regressor, rf_classifier, X_train, X_test, feature_names
#     )
    
#     # Get top features for CMC prediction
#     reg_feature_importances = rf_regressor.feature_importances_
#     top_cmc_features = np.argsort(reg_feature_importances)[::-1]
#     print("\nTop 10 features for CMC prediction:")
#     for i in range(10):
#         idx = top_cmc_features[i]
#         print(f"{feature_names[idx]}: {reg_feature_importances[idx]:.4f}")
    
#     # Get top features for cluster classification
#     cls_feature_importances = rf_classifier.feature_importances_
#     top_cluster_features = np.argsort(cls_feature_importances)[::-1]
#     print("\nTop 10 features for cluster classification:")
#     for i in range(10):
#         idx = top_cluster_features[i]
#         print(f"{feature_names[idx]}: {cls_feature_importances[idx]:.4f}")
    
#     # Plot feature distributions by cluster
#     cluster_dist_plot = plot_feature_distributions_by_cluster(
#         df, X, top_cluster_features, feature_names
#     )
    
#     # Save plots
#     for name, plot in plots.items():
#         filename = f"shap_{name}.png"
#         plot.savefig(filename, dpi=300, bbox_inches='tight')
#         print(f"Saved {filename}")
    
#     cluster_dist_plot.savefig("feature_distributions_by_cluster.png", dpi=300, bbox_inches='tight')
#     print("Saved feature_distributions_by_cluster.png")
    
#     # Show plots
#     plt.show()

# if __name__ == "__main__":
#     main()