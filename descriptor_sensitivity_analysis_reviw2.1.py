"""
Comprehensive Descriptor Sensitivity Analysis for Clustering
Addresses Reviewer Comment: Justify descriptor choice and assess clustering sensitivity

This script performs:
1. Alternative descriptor set comparisons (minimal, extended, fingerprints)
2. Alternative dimensionality reduction methods (PCA, t-SNE, UMAP)
3. Alternative clustering methods (K-means, hierarchical, HDBSCAN)
4. VIF analysis for multicollinearity
5. Correlation matrix analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr
import pacmap
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: HDBSCAN not available. Install with: pip install hdbscan")

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import datetime
import warnings
warnings.filterwarnings('ignore')


class DescriptorSensitivityAnalyzer:
    """Comprehensive analysis of descriptor choice and clustering sensitivity"""
    
    def __init__(self, data_path, results_folder=None):
        """
        Initialize analyzer
        
        Parameters:
        -----------
        data_path : str
            Path to CSV file with molecular descriptors
        results_folder : str, optional
            Custom output directory
        """
        self.data_path = data_path
        
        if results_folder is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.results_folder = f"Results/descriptor_sensitivity/{timestamp}"
        else:
            self.results_folder = results_folder
            
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"Results will be saved to: {self.results_folder}")
        
        # Load and prepare data
        self.df_original = pd.read_csv(data_path)
        self.prepare_data()
        
        # Storage for results
        self.clustering_results = {}
        self.metrics_df = None
        
    def prepare_data(self):
        """Prepare data by separating features from metadata"""
        self.df = self.df_original.copy()
        
        # Store metadata
        self.metadata_cols = []
        for col in ['smiles', 'TARGET', 'Index', 'cmc', 'Log(cmc*1000+1)']:
            if col in self.df.columns:
                self.metadata_cols.append(col)
        
        # Store SMILES if available
        if 'smiles' in self.df.columns:
            self.smiles = self.df['smiles'].values
        else:
            self.smiles = None
            
        # Store target if available
        if 'TARGET' in self.df.columns:
            self.target = self.df['TARGET'].values
        else:
            self.target = None
        
        # Get feature columns (exclude metadata)
        self.feature_cols = [col for col in self.df.columns if col not in self.metadata_cols]
        self.X_full = self.df[self.feature_cols].values
        
        print(f"Data loaded: {self.X_full.shape[0]} molecules, {self.X_full.shape[1]} descriptors")
        print(f"Feature columns: {self.feature_cols}")
        
    def create_alternative_descriptor_sets(self):
        """Create alternative descriptor sets for comparison"""
        descriptor_sets = {}
        
        # SET A: Original 25 descriptors (baseline)
        descriptor_sets['Original_25'] = {
            'features': self.feature_cols,
            'data': self.X_full,
            'description': 'Original 25 RDKit descriptors'
        }
        
        # SET B: Minimal 10 descriptors (most important from literature)
        # Based on surfactant QSPR studies - focus on amphiphilic balance
        minimal_descriptors = [
            'MW', 'LogP', 'TPSA', 'HBA', 'HBD',
            'LongestAliphaticChain', 'NumCarbonyls', 'NumChargedAtoms',
            'PolarRatio', 'FractionCSP3'
        ]
        minimal_descriptors = [d for d in minimal_descriptors if d in self.feature_cols]
        
        descriptor_sets['Minimal_10'] = {
            'features': minimal_descriptors,
            'data': self.df[minimal_descriptors].values,
            'description': 'Minimal 10 surfactant-relevant descriptors'
        }
        
        # SET C: Extended descriptors (if more are available or compute additional)
        # For now, we'll use all available descriptors
        descriptor_sets['Extended'] = {
            'features': self.feature_cols,
            'data': self.X_full,
            'description': 'Extended descriptor set (same as original for this dataset)'
        }
        
        # SET D: Feature subsets by category
        molecular_props = ['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotBonds', 'MolMR', 'LabuteASA']
        structural = ['NumAromRings', 'FractionCSP3', 'NumRings', 'HeavyAtomCount', 
                     'CarbonCount', 'AliphaticCarbonCount', 'LongestAliphaticChain']
        functional = ['NumOH', 'NumCarbonyls', 'NumCarboxyls', 'NumEthers', 
                     'NumSulfurs', 'NumAmines', 'NumChargedAtoms']
        
        molecular_props = [d for d in molecular_props if d in self.feature_cols]
        structural = [d for d in structural if d in self.feature_cols]
        functional = [d for d in functional if d in self.feature_cols]
        
        descriptor_sets['MolecularProps_Only'] = {
            'features': molecular_props,
            'data': self.df[molecular_props].values,
            'description': 'Molecular properties only'
        }
        
        descriptor_sets['Structural_Only'] = {
            'features': structural,
            'data': self.df[structural].values,
            'description': 'Structural features only'
        }
        
        descriptor_sets['Functional_Only'] = {
            'features': functional,
            'data': self.df[functional].values,
            'description': 'Functional groups only'
        }
        
        # SET E: Fingerprint-based features (if SMILES available)
        if self.smiles is not None:
            print("Computing molecular fingerprints...")
            maccs_fps = self.compute_maccs_fingerprints()
            morgan_fps = self.compute_morgan_fingerprints()
            
            descriptor_sets['MACCS_Keys'] = {
                'features': [f'MACCS_{i}' for i in range(maccs_fps.shape[1])],
                'data': maccs_fps,
                'description': 'MACCS structural keys (167 bits)'
            }
            
            descriptor_sets['Morgan_FP'] = {
                'features': [f'Morgan_{i}' for i in range(morgan_fps.shape[1])],
                'data': morgan_fps,
                'description': 'Morgan fingerprints (1024 bits, radius=2)'
            }
        
        self.descriptor_sets = descriptor_sets
        return descriptor_sets
    
    def compute_maccs_fingerprints(self):
        """Compute MACCS keys fingerprints"""
        fps = []
        for smi in self.smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fps.append(list(fp))
            else:
                fps.append([0] * 167)  # MACCS has 167 bits
        return np.array(fps)
    
    def compute_morgan_fingerprints(self, radius=2, n_bits=1024):
        """Compute Morgan (circular) fingerprints"""
        fps = []
        for smi in self.smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(list(fp))
            else:
                fps.append([0] * n_bits)
        return np.array(fps)
    
    def perform_clustering_analysis(self, descriptor_set_name='Original_25'):
        """
        Perform comprehensive clustering analysis on a descriptor set
        
        Parameters:
        -----------
        descriptor_set_name : str
            Name of descriptor set to analyze
        """
        print(f"\n{'='*70}")
        print(f"Analyzing: {descriptor_set_name}")
        print(f"{'='*70}")
        
        desc_set = self.descriptor_sets[descriptor_set_name]
        X = desc_set['data']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {
            'descriptor_set': descriptor_set_name,
            'n_features': X.shape[1],
            'description': desc_set['description']
        }
        
        # Test multiple dimensionality reduction methods
        dr_methods = {
            'PaCMAP': self.apply_pacmap,
            'PCA': self.apply_pca,
            't-SNE': self.apply_tsne,
        }
        
        if UMAP_AVAILABLE:
            dr_methods['UMAP'] = self.apply_umap
        
        # Test multiple clustering methods
        clustering_methods = {
            'DBSCAN': self.apply_dbscan,
            'KMeans_4': lambda X: self.apply_kmeans(X, n_clusters=4),
            'KMeans_6': lambda X: self.apply_kmeans(X, n_clusters=6),
            'Hierarchical_4': lambda X: self.apply_hierarchical(X, n_clusters=4),
        }
        
        if HDBSCAN_AVAILABLE:
            clustering_methods['HDBSCAN'] = self.apply_hdbscan
        
        # Perform all combinations
        for dr_name, dr_func in dr_methods.items():
            print(f"  Dimensionality reduction: {dr_name}")
            X_reduced = dr_func(X_scaled)
            
            for clust_name, clust_func in clustering_methods.items():
                labels = clust_func(X_reduced)
                
                # Compute metrics
                metrics = self.compute_clustering_metrics(X_reduced, labels)
                
                # Store results
                key = f"{descriptor_set_name}_{dr_name}_{clust_name}"
                results[key] = {
                    'labels': labels,
                    'X_reduced': X_reduced,
                    'metrics': metrics
                }
                
                print(f"    {clust_name}: {metrics['n_clusters']} clusters, "
                      f"Silhouette={metrics['silhouette']:.3f}")
        
        self.clustering_results[descriptor_set_name] = results
        return results
    
    def apply_pacmap(self, X):
        """Apply PaCMAP dimensionality reduction"""
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.6,
                               FP_ratio=1.5, num_iters=200, random_state=101)
        return reducer.fit_transform(X)
    
    def apply_pca(self, X):
        """Apply PCA dimensionality reduction"""
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X)
    
    def apply_tsne(self, X):
        """Apply t-SNE dimensionality reduction"""
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        return tsne.fit_transform(X)
    
    def apply_umap(self, X):
        """Apply UMAP dimensionality reduction"""
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        return reducer.fit_transform(X)
    
    def apply_dbscan(self, X):
        """Apply DBSCAN clustering"""
        dbscan = DBSCAN(eps=3, min_samples=5)
        return dbscan.fit_predict(X)
    
    def apply_hdbscan(self, X):
        """Apply HDBSCAN clustering"""
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        return clusterer.fit_predict(X)
    
    def apply_kmeans(self, X, n_clusters=4):
        """Apply K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(X)
    
    def apply_hierarchical(self, X, n_clusters=4):
        """Apply hierarchical clustering"""
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return clustering.fit_predict(X)
    
    def compute_clustering_metrics(self, X, labels):
        """Compute clustering quality metrics"""
        # Filter out noise points (label = -1) for metrics
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        n_clusters = len(np.unique(labels_filtered))
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': np.sum(labels == -1),
            'noise_ratio': np.sum(labels == -1) / len(labels)
        }
        
        # Only compute if we have valid clusters
        if n_clusters > 1 and len(labels_filtered) > n_clusters:
            try:
                metrics['silhouette'] = silhouette_score(X_filtered, labels_filtered)
            except:
                metrics['silhouette'] = np.nan
            
            try:
                metrics['davies_bouldin'] = davies_bouldin_score(X_filtered, labels_filtered)
            except:
                metrics['davies_bouldin'] = np.nan
        else:
            metrics['silhouette'] = np.nan
            metrics['davies_bouldin'] = np.nan
        
        return metrics
    
    def compare_to_baseline(self, baseline_name='Original_25'):
        """
        Compare all clustering results to baseline using Adjusted Rand Index
        
        Parameters:
        -----------
        baseline_name : str
            Name of baseline descriptor set
        """
        print(f"\n{'='*70}")
        print(f"Computing Adjusted Rand Index vs. Baseline ({baseline_name})")
        print(f"{'='*70}")
        
        # Get baseline clustering (PaCMAP + DBSCAN)
        baseline_key = f"{baseline_name}_PaCMAP_DBSCAN"
        baseline_labels = self.clustering_results[baseline_name][baseline_key]['labels']
        
        comparison_results = []
        
        for desc_name, results in self.clustering_results.items():
            for key in results.keys():
                if key not in ['descriptor_set', 'n_features', 'description']:
                    labels = results[key]['labels']
                    ari = adjusted_rand_score(baseline_labels, labels)
                    
                    parts = key.split('_')
                    dr_method = parts[-2]
                    clust_method = parts[-1] if len(parts) == 4 else '_'.join(parts[-2:])
                    
                    comparison_results.append({
                        'Descriptor_Set': desc_name,
                        'DR_Method': dr_method,
                        'Clustering': clust_method,
                        'ARI_vs_Baseline': ari,
                        'N_Clusters': results[key]['metrics']['n_clusters'],
                        'Silhouette': results[key]['metrics']['silhouette'],
                        'Davies_Bouldin': results[key]['metrics']['davies_bouldin']
                    })
        
        self.comparison_df = pd.DataFrame(comparison_results)
        
        # Save to CSV
        output_path = os.path.join(self.results_folder, 'descriptor_sensitivity_results.csv')
        self.comparison_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        return self.comparison_df
    
    def compute_vif_analysis(self):
        """Compute Variance Inflation Factor for multicollinearity detection"""
        print(f"\n{'='*70}")
        print("Computing VIF Analysis")
        print(f"{'='*70}")
        
        X = self.X_full
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        vif_data = []
        for i, feature in enumerate(self.feature_cols):
            try:
                vif = variance_inflation_factor(X_scaled, i)
                vif_data.append({'Feature': feature, 'VIF': vif})
            except:
                vif_data.append({'Feature': feature, 'VIF': np.nan})
        
        self.vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
        
        # Save to CSV
        output_path = os.path.join(self.results_folder, 'vif_analysis.csv')
        self.vif_df.to_csv(output_path, index=False)
        print(f"VIF analysis saved to: {output_path}")
        
        return self.vif_df
    
    def compute_correlation_analysis(self):
        """Compute correlation matrix of descriptors"""
        print(f"\n{'='*70}")
        print("Computing Correlation Analysis")
        print(f"{'='*70}")
        
        X = self.X_full
        self.corr_matrix = pd.DataFrame(X, columns=self.feature_cols).corr()
        
        # Save to CSV
        output_path = os.path.join(self.results_folder, 'correlation_matrix.csv')
        self.corr_matrix.to_csv(output_path)
        print(f"Correlation matrix saved to: {output_path}")
        
        return self.corr_matrix
    
    def plot_clustering_robustness(self):
        """Create comprehensive visualization of clustering robustness"""
        if self.comparison_df is None:
            print("Run compare_to_baseline() first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: ARI heatmap by descriptor set and method
        pivot_ari = self.comparison_df.pivot_table(
            values='ARI_vs_Baseline',
            index='Descriptor_Set',
            columns='DR_Method',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_ari, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0.5, vmin=0, vmax=1, ax=axes[0, 0])
        axes[0, 0].set_title('Adjusted Rand Index vs. Baseline\n(Higher = More Similar)', fontsize=12)
        axes[0, 0].set_xlabel('Dimensionality Reduction Method')
        axes[0, 0].set_ylabel('Descriptor Set')
        
        # Plot 2: Silhouette scores by descriptor set
        pivot_sil = self.comparison_df.pivot_table(
            values='Silhouette',
            index='Descriptor_Set',
            columns='DR_Method',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_sil, annot=True, fmt='.3f', cmap='viridis',
                   ax=axes[0, 1])
        axes[0, 1].set_title('Silhouette Score\n(Higher = Better Separation)', fontsize=12)
        axes[0, 1].set_xlabel('Dimensionality Reduction Method')
        axes[0, 1].set_ylabel('Descriptor Set')
        
        # Plot 3: ARI distribution by descriptor set
        desc_order = self.comparison_df.groupby('Descriptor_Set')['ARI_vs_Baseline'].mean().sort_values(ascending=False).index
        
        sns.boxplot(data=self.comparison_df, y='Descriptor_Set', x='ARI_vs_Baseline',
                   order=desc_order, ax=axes[1, 0])
        axes[1, 0].axvline(0.8, color='green', linestyle='--', label='High similarity (ARI > 0.8)')
        axes[1, 0].axvline(0.5, color='orange', linestyle='--', label='Moderate similarity')
        axes[1, 0].set_title('ARI Distribution Across All Methods', fontsize=12)
        axes[1, 0].set_xlabel('Adjusted Rand Index')
        axes[1, 0].legend()
        
        # Plot 4: Number of clusters identified
        pivot_clust = self.comparison_df.pivot_table(
            values='N_Clusters',
            index='Descriptor_Set',
            columns='Clustering',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_clust, annot=True, fmt='.1f', cmap='coolwarm',
                   ax=axes[1, 1])
        axes[1, 1].set_title('Number of Clusters Identified', fontsize=12)
        axes[1, 1].set_xlabel('Clustering Method')
        axes[1, 1].set_ylabel('Descriptor Set')
        
        plt.tight_layout()
        output_path = os.path.join(self.results_folder, 'clustering_robustness_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Robustness plot saved to: {output_path}")
        plt.close()
    
    def plot_correlation_matrix(self):
        """Visualize correlation matrix of descriptors"""
        if self.corr_matrix is None:
            self.compute_correlation_analysis()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))
        
        sns.heatmap(self.corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Correlation Matrix of Molecular Descriptors', fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(self.results_folder, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix plot saved to: {output_path}")
        plt.close()
        
        # Also create a plot showing high correlations
        self.plot_high_correlations()
    
    def plot_high_correlations(self, threshold=0.8):
        """Plot pairs of descriptors with high correlation"""
        if self.corr_matrix is None:
            return
        
        # Find high correlations (excluding diagonal)
        corr_unstacked = self.corr_matrix.unstack()
        high_corr = corr_unstacked[
            (corr_unstacked.abs() > threshold) &
            (corr_unstacked.abs() < 1.0)
        ].sort_values(ascending=False)
        
        # Remove duplicates (A-B and B-A are the same)
        seen = set()
        unique_pairs = []
        for idx, val in high_corr.items():
            pair = tuple(sorted(idx))
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append((idx[0], idx[1], val))
        
        if len(unique_pairs) > 0:
            fig, ax = plt.subplots(figsize=(10, max(6, len(unique_pairs) * 0.4)))
            
            features = [f"{p[0]} ↔ {p[1]}" for p in unique_pairs]
            correlations = [p[2] for p in unique_pairs]
            
            colors = ['red' if c > 0 else 'blue' for c in correlations]
            ax.barh(features, correlations, color=colors, alpha=0.7)
            ax.axvline(threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
            ax.axvline(-threshold, color='orange', linestyle='--')
            ax.set_xlabel('Correlation Coefficient', fontsize=12)
            ax.set_title(f'Highly Correlated Descriptor Pairs (|r| > {threshold})', fontsize=14)
            ax.legend()
            plt.tight_layout()
            
            output_path = os.path.join(self.results_folder, 'high_correlations.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"High correlations plot saved to: {output_path}")
            plt.close()
    
    def plot_vif_analysis(self):
        """Visualize VIF analysis results"""
        if self.vif_df is None:
            self.compute_vif_analysis()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by VIF
        vif_sorted = self.vif_df.sort_values('VIF', ascending=True)
        
        colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' 
                 for v in vif_sorted['VIF']]
        
        ax.barh(vif_sorted['Feature'], vif_sorted['VIF'], color=colors, alpha=0.7)
        ax.axvline(5, color='orange', linestyle='--', label='Moderate multicollinearity (VIF=5)')
        ax.axvline(10, color='red', linestyle='--', label='High multicollinearity (VIF=10)')
        ax.set_xlabel('Variance Inflation Factor', fontsize=12)
        ax.set_title('Multicollinearity Analysis (VIF)', fontsize=14)
        ax.legend()
        plt.tight_layout()
        
        output_path = os.path.join(self.results_folder, 'vif_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"VIF plot saved to: {output_path}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate text summary of analysis"""
        report = []
        report.append("="*70)
        report.append("DESCRIPTOR SENSITIVITY ANALYSIS - SUMMARY REPORT")
        report.append("="*70)
        report.append("")
        
        # Dataset info
        report.append(f"Dataset: {self.data_path}")
        report.append(f"Number of molecules: {self.X_full.shape[0]}")
        report.append(f"Number of descriptors (original): {self.X_full.shape[1]}")
        report.append("")
        
        # Clustering robustness
        if self.comparison_df is not None:
            report.append("CLUSTERING ROBUSTNESS:")
            report.append("-" * 70)
            
            # Summary by descriptor set
            summary = self.comparison_df.groupby('Descriptor_Set').agg({
                'ARI_vs_Baseline': ['mean', 'std', 'min', 'max'],
                'Silhouette': ['mean', 'std']
            }).round(3)
            
            report.append("\nAverage ARI vs. Baseline by Descriptor Set:")
            for desc_set in summary.index:
                ari_mean = summary.loc[desc_set, ('ARI_vs_Baseline', 'mean')]
                ari_std = summary.loc[desc_set, ('ARI_vs_Baseline', 'std')]
                report.append(f"  {desc_set:25s}: {ari_mean:.3f} ± {ari_std:.3f}")
            
            report.append("")
            report.append("Interpretation:")
            report.append("  ARI > 0.8: High similarity (clustering is robust)")
            report.append("  ARI 0.5-0.8: Moderate similarity")
            report.append("  ARI < 0.5: Low similarity (descriptor choice matters)")
            report.append("")
        
        # Multicollinearity
        if self.vif_df is not None:
            report.append("MULTICOLLINEARITY ANALYSIS (VIF):")
            report.append("-" * 70)
            high_vif = self.vif_df[self.vif_df['VIF'] > 10]
            if len(high_vif) > 0:
                report.append(f"Features with VIF > 10 (high multicollinearity): {len(high_vif)}")
                for _, row in high_vif.iterrows():
                    report.append(f"  {row['Feature']:25s}: VIF = {row['VIF']:.2f}")
            else:
                report.append("No features with high multicollinearity (VIF > 10)")
            report.append("")
        
        # High correlations
        if self.corr_matrix is not None:
            report.append("HIGH CORRELATIONS:")
            report.append("-" * 70)
            corr_unstacked = self.corr_matrix.unstack()
            high_corr = corr_unstacked[
                (corr_unstacked.abs() > 0.8) &
                (corr_unstacked.abs() < 1.0)
            ]
            report.append(f"Number of highly correlated pairs (|r| > 0.8): {len(high_corr) // 2}")
            report.append("")
        
        # Save report
        report_text = "\n".join(report)
        output_path = os.path.join(self.results_folder, 'summary_report.txt')
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\nSummary report saved to: {output_path}")
        
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete sensitivity analysis pipeline"""
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE DESCRIPTOR SENSITIVITY ANALYSIS")
        print("="*70 + "\n")
        
        # Step 1: Create alternative descriptor sets
        print("Step 1: Creating alternative descriptor sets...")
        self.create_alternative_descriptor_sets()
        print(f"Created {len(self.descriptor_sets)} descriptor sets")
        
        # Step 2: Perform clustering analysis on each set
        print("\nStep 2: Performing clustering analysis...")
        for desc_name in self.descriptor_sets.keys():
            self.perform_clustering_analysis(desc_name)
        
        # Step 3: Compare to baseline
        print("\nStep 3: Comparing to baseline...")
        self.compare_to_baseline()
        
        # Step 4: VIF analysis
        print("\nStep 4: Computing VIF analysis...")
        self.compute_vif_analysis()
        
        # Step 5: Correlation analysis
        print("\nStep 5: Computing correlation analysis...")
        self.compute_correlation_analysis()
        
        # Step 6: Generate visualizations
        print("\nStep 6: Generating visualizations...")
        self.plot_clustering_robustness()
        self.plot_correlation_matrix()
        self.plot_vif_analysis()
        
        # Step 7: Generate summary report
        print("\nStep 7: Generating summary report...")
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print(f"All results saved to: {self.results_folder}")
        print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage for Data1
    print("Running analysis for Data1...")
    analyzer1 = DescriptorSensitivityAnalyzer(
        data_path="DATA/valid_molecules_data_cmc.csv",
        results_folder="Results/descriptor_sensitivity_rev2.1/data1"
    )
    analyzer1.run_complete_analysis()
    
    # Example usage for Data2
    print("\n\nRunning analysis for Data2...")
    analyzer2 = DescriptorSensitivityAnalyzer(
        data_path="DATA/valid_molecules_data_cmc_data2.csv",
        results_folder="Results/descriptor_sensitivity_rev2.1/data2"
    )
    analyzer2.run_complete_analysis()
    
    print("\n✓ All analyses complete!")
