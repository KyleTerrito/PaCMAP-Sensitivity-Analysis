import numpy as np
import pandas as pd
import sompy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import sqrt
import scipy
from matplotlib.colors import ListedColormap

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
import datetime
from sklearn.preprocessing import StandardScaler

# Create results directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f"Results_SOMPY_PacMAP/SOMPY_PacMAP_{timestamp}"
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

# Load PacMAP cluster data
input_folder = r"RESULTS/2025-03-09_20-03-29-logcmc"
pacmap_clusters = pd.read_csv(os.path.join(input_folder, "cluster_labels.csv"))["Cluster Labels"]
print(f"Loaded {len(pacmap_clusters)} PacMAP cluster labels")
print(f"Unique PacMAP clusters: {sorted(pacmap_clusters.unique())}")

# Define SOM parameters
som_l, som_w = 30, 24

class UMatrixViewNew(sompy.umatrix.UMatrixView):
    def Get_label_Umatrix(self, class_labels, coord):
        map_w, map_l = max(coord[:,1]) + 1, max(coord[:,0])+1
        label_set = list(set(class_labels))
        label_temp = np.zeros(len(label_set))-1
        label_matrix = np.zeros((map_l, map_w))-1

        for xc in range(map_w):
            for yc in range(map_l):
                label_temp = np.zeros(len(label_set))-1
                for i in range(len(coord)):
                    if coord[i,1] == xc and coord[i,0] == yc:
                        label_ = class_labels[i]
                        index = label_set.index(label_)
                        label_temp[index] = label_temp[index] + 1
                        
                if max(label_temp) != -1:
                    label_temp = list(label_temp)
                    take_all_index = label_temp.index(max(label_temp))
                    label_matrix[yc, xc] = take_all_index

        return label_matrix 
        
    def set_fig_ax(self,fig, ax):
        self.fig = fig
        self.ax = ax
        
    def Sigmoid(self, umat):
        l,w = np.shape(umat)
        max_u= np.max(umat)
        for i in range(l):
            for j in range(w):
                x= umat[i,j]
                umat[i,j]= (1 / (1 + np.exp(-x)) - 0.5) *2 * max_u
        
    def show(self, som, distance2=1, row_normalized=False, show_data=False,
            contooor=False, blob=False, labels=False, show_troj=False, 
            troj_range=False, troj_color = None, troj_marker = None,
            show_mag = False,show_class = False, class_labels = None, color_map=None):
        umat = self.build_u_matrix(som, distance=distance2,
                                row_normalized=row_normalized)
        msz = som.codebook.mapsize
        proj = som.project_data(som.data_raw)
        coord = som.bmu_ind_to_xy(proj)
            
        fig = self.fig
        ax = self.ax
        if show_troj==True and len(troj_range)>1:
            
            if troj_color is None:			  
                ax.plot(coord[troj_range, 1], coord[troj_range, 0], '-w',linewidth=1)
                ax.plot(coord[troj_range, 1], coord[troj_range, 0], '.w',linewidth=1)
                
            else:
                ax.scatter(coord[troj_range, 1], coord[troj_range, 0],c=troj_color, marker=troj_marker ,s=10, alpha = 0.8, edgecolors = 'w')
            
        if contooor:
            mn = np.min(umat.flatten())
            mx = np.max(umat.flatten())
            std = np.std(umat.flatten())
            md = np.median(umat.flatten())
            mx = md + 0*std
            ax.contour(umat, np.linspace(mn, mx, 15), linewidths=0.7,
                    cmap=plt.cm.get_cmap('Blues'))
            
        if show_data:
            ax.scatter(coord[:, 1], coord[:, 0], s=2, alpha=1., c='Gray',
                        marker='o', cmap='viridis', linewidths=3, edgecolor='Gray')
            ax.axis('off')
            
        if labels:
            if labels is True:
                labels = som.build_data_labels()
            for label, x, y in zip(labels, coord[:, 1], coord[:, 0]):
                plt.annotate(str(label), xy=(x, y),
                            horizontalalignment='center',
                            verticalalignment='center')
            
        sel_points = list()
            
        if blob:
            from skimage.color import rgb2gray
            from skimage.feature import blob_log
                
            image = 1 / umat
            rgb2gray(image)
                
            # 'Laplacian of Gaussian'
            blobs = blob_log(image, max_sigma=5, num_sigma=4, threshold=.152)
            blobs[:, 2] = blobs[:, 2] * sqrt(2)
            sel_points = list()
                
            for blob in blobs:
                row, col, r = blob
                c = plt.Circle((col, row), r, color='red', linewidth=2,
                            fill=False)
                ax.add_patch(c)
                dist = scipy.spatial.distance_matrix(
                    coord[:, :2], np.array([row, col])[np.newaxis, :])
                sel_point = dist <= r
                ax.plot(coord[:, 1][sel_point[:, 0]],
                        coord[:, 0][sel_point[:, 0]], '.r')
                sel_points.append(sel_point[:, 0])
                
        if show_class == True and class_labels is not None:
            umat_class = self.Get_label_Umatrix(class_labels, coord)
            im = ax.imshow(umat_class, interpolation='nearest', cmap=color_map, alpha=1)
        else:
            self.Sigmoid(umat)
            # Use the provided color_map if given, otherwise default to jet
            if color_map is None:
                color_map = 'jet'
            
            im = ax.imshow(umat, interpolation='nearest', cmap=plt.colormaps[color_map], alpha=1)

        if show_mag:
            divider = make_axes_locatable(ax)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax1)
            cb.ax.tick_params(labelsize=10)
            
        return sel_points, umat

class SOM_Tool():
    def __init__(self, nrm_dt, mapsize=(30,24)):
        self.dataf = nrm_dt
        self.map_size = mapsize
    
    def create_title(self, method, length):
        title = []        
        for i in range(length):
            piece = str(i+1) + ". " + method + str(i+1)
            title.append(piece)
        return title
    
    def Do_SOM(self):
        self.som = sompy.SOMFactory.build(np.array(self.dataf), self.map_size, mask=None, mapshape="planar", lattice="rect", normalization="var", initialization="pca", neighborhood="gaussian", training="batch", name="sompy")  
        self.som.train(n_job=1, verbose="info")  # verbose="debug" will print more, and verbose=None wont print anything
    
    def SOM_DR(self):        
        DR_data = self.som._normalizer.normalize_by(self.som.data_raw, self.som.codebook.matrix)       
        idx = range(len(DR_data))
        columns = self.create_title("SOM", len(DR_data[0]))
        df = pd.DataFrame(DR_data, index=idx, columns=columns)
        return df
        
    def Draw_SOM_RAW(self, fig=None, ax=None, contooor=False, blob=False, show_data=False, 
                    show_troj=False, troj_range=None, show_mag=True, color_map='jet'):      
        self.fig = fig
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        visual = UMatrixViewNew(1, 1, "", col_size=200)
        visual.set_fig_ax(self.fig, self.ax)
        sel_points, umat = visual.show(self.som, contooor=contooor, blob=blob, show_data=show_data,
                         show_troj=show_troj, troj_range=troj_range, show_mag=show_mag, color_map=color_map)
        
        # Get the coordinates after visualization
        proj = self.som.project_data(self.som.data_raw)
        coord = self.som.bmu_ind_to_xy(proj)
        return coord
        
    def Draw_SOM_Clusters(self, class_labels, fig=None, ax=None, show_mag=True, color_map='tab10'):
        self.fig = fig
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        visual = UMatrixViewNew(1, 1, "", col_size=200)
        visual.set_fig_ax(self.fig, self.ax)
        sel_points, umat = visual.show(self.som, show_class=True, class_labels=class_labels, show_mag=show_mag, color_map=color_map)
        
        # Get the coordinates after visualization
        proj = self.som.project_data(self.som.data_raw)
        coord = self.som.bmu_ind_to_xy(proj)
        return coord

# ---------------------- EXECUTION SECTION ----------------------
if __name__ == "__main__":
    # Use the scaled data for SOM
    nrm_df = data_scaled
    
    # Create and train the SOM
    print("Training SOM...")
    som_class = SOM_Tool(nrm_df, mapsize=(som_l, som_w))
    som_class.Do_SOM()
    DR_df = som_class.SOM_DR()
    print(f"{som_l} x {som_w} was used to generate SOMPY")
    
    # Create the raw SOM visualization and save it
    print("Creating raw SOM visualization...")
    fig_som = plt.figure(figsize=(10, 7))
    coord = som_class.Draw_SOM_RAW(fig=fig_som, show_mag=True)
    plt.title("SOM U-Matrix")
    plt.tight_layout()
    
    # Save raw SOM to the results folder
    output_path = os.path.join(results_folder, "raw_som.png")
    plt.savefig(output_path)
    print(f"Raw SOM visualization saved to {output_path}")
    plt.close()
    
    # Create SOM with PacMAP clusters and save it
    print("Creating SOM with PacMAP clusters...")
    fig_som_pacmap = plt.figure(figsize=(10, 7))
    coord = som_class.Draw_SOM_Clusters(pacmap_clusters, fig=fig_som_pacmap, show_mag=True, color_map='tab10')
    plt.title("SOM with PacMAP Clusters")
    plt.tight_layout()
    
    # Save cluster SOM to the results folder
    output_path = os.path.join(results_folder, "som_pacmap_clusters.png")
    plt.savefig(output_path)
    print(f"SOM with PacMAP clusters saved to {output_path}")
    plt.close()
    
    # Create individual projection plots for each cluster
    print("Creating individual cluster projections...")
    unique_clusters = sorted(pacmap_clusters.unique())
    
    # Get the U-matrix and apply Sigmoid transformation for consistent visualization
    visual = UMatrixViewNew(1, 1, "", col_size=200)
    umat = visual.build_u_matrix(som_class.som)
    
    # Apply the same Sigmoid transformation used in the original visualization
    l, w = np.shape(umat)
    max_u = np.max(umat)
    for i in range(l):
        for j in range(w):
            x = umat[i, j]
            umat[i, j] = (1 / (1 + np.exp(-x)) - 0.5) * 2 * max_u
    
    # Project data to SOM
    proj = som_class.som.project_data(som_class.som.data_raw)
    bmu_coord = som_class.som.bmu_ind_to_xy(proj)
    
    for cluster in unique_clusters:
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(umat, cmap='jet', interpolation='none')
        plt.colorbar(label='Distance')
        
        # Find data points belonging to this cluster
        cluster_indices = np.where(pacmap_clusters == cluster)[0]
        
        # Plot these points on the map
        cluster_coords = bmu_coord[cluster_indices]
        plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0], 
                    color='white', marker='o', s=50, edgecolors='black')
        
        plt.title(f"Projection of Cluster {cluster}")
        plt.tight_layout()
        
        # Save individual cluster projection
        cluster_output_path = os.path.join(results_folder, f"cluster_{cluster}_projection.png")
        plt.savefig(cluster_output_path)
        print(f"Cluster {cluster} projection saved to {cluster_output_path}")
        plt.close()
    
    for cluster in unique_clusters:
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(umat, cmap='jet', interpolation='none')
        plt.colorbar(label='Distance')
        
        # Find data points belonging to this cluster
        cluster_indices = np.where(pacmap_clusters == cluster)[0]
        
        # Plot these points on the map
        cluster_coords = bmu_coord[cluster_indices]
        plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0], 
                    color='white', marker='o', s=50, edgecolors='black')
        
        plt.title(f"Projection of Cluster {cluster}")
        plt.tight_layout()
        
        # Save individual cluster projection
        cluster_output_path = os.path.join(results_folder, f"cluster_{cluster}_projection.png")
        plt.savefig(cluster_output_path)
        print(f"Cluster {cluster} projection saved to {cluster_output_path}")
        plt.close()
    
    # Save the SOM projection with cluster labels
    print("Creating SOM data with PacMAP clusters...")
    
    # Project data to SOM
    proj = som_class.som.project_data(som_class.som.data_raw)
    bmu_coord = som_class.som.bmu_ind_to_xy(proj)
    
    # Create a DataFrame with SOM projection and PacMAP cluster labels
    som_proj_df = pd.DataFrame({
        'BMU_X': bmu_coord[:, 0],
        'BMU_Y': bmu_coord[:, 1],
        'PacMAP_Cluster': pacmap_clusters.values
    })
    
    # Add original names and targets if available
    if len(names) == len(som_proj_df):
        som_proj_df['Name'] = names.values
    if len(targets) == len(som_proj_df):
        som_proj_df['Target'] = targets.values
    
    # Save the combined data
    combined_output_path = os.path.join(results_folder, "som_pacmap_data.csv")
    som_proj_df.to_csv(combined_output_path, index=False)
    print(f"SOM data with PacMAP clusters saved to {combined_output_path}")
    
    # Save the DR dataframe
    dr_output_path = os.path.join(results_folder, "som_dr_data.csv")
    DR_df.to_csv(dr_output_path)
    print(f"SOM dimensionality reduction data saved to {dr_output_path}")
    
    # Create a visualization showing the distribution of PacMAP clusters on the SOM grid
    print("Creating cluster distribution analysis...")
    
    # Count the occurrences of each cluster in each SOM cell
    cluster_distribution = {}
    unique_clusters = sorted(pacmap_clusters.unique())
    
    for cluster in unique_clusters:
        cluster_indices = np.where(pacmap_clusters == cluster)[0]
        cluster_distribution[f"Cluster_{cluster}"] = len(cluster_indices)
    
    # Create a bar chart showing cluster sizes
    fig_distribution = plt.figure(figsize=(12, 6))
    plt.bar(cluster_distribution.keys(), cluster_distribution.values())
    plt.title("PacMAP Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the distribution chart
    dist_output_path = os.path.join(results_folder, "pacmap_cluster_distribution.png")
    plt.savefig(dist_output_path)
    print(f"Cluster distribution chart saved to {dist_output_path}")
    plt.close()
    
    # Analyze cluster overlaps in SOM grid
    print("Analyzing cluster-SOM grid overlaps...")
    
    # Group by SOM grid cell and count clusters in each cell
    som_cells = {}
    for i in range(len(bmu_coord)):
        cell = (int(bmu_coord[i, 0]), int(bmu_coord[i, 1]))
        cluster = pacmap_clusters.iloc[i]
        
        if cell not in som_cells:
            som_cells[cell] = []
        
        som_cells[cell].append(cluster)
    
    # Count cells with single vs multiple clusters
    pure_cells = 0
    mixed_cells = 0
    for cell, clusters in som_cells.items():
        unique_in_cell = len(set(clusters))
        if unique_in_cell == 1:
            pure_cells += 1
        else:
            mixed_cells += 1
    
    print(f"SOM-PacMAP Analysis:")
    print(f"- Pure cells (single cluster): {pure_cells}")
    print(f"- Mixed cells (multiple clusters): {mixed_cells}")
    print(f"- Total used cells: {len(som_cells)}")
    print(f"- Grid purity: {pure_cells/len(som_cells)*100:.1f}%")
    
    # Create a list of clusters and their cell assignments
    # (matching the format from your example)
    index_order = sorted(unique_clusters)
    print(f"Index order: {index_order}")
    
    # Create groups of indices for each cluster
    group_selected = []
    for label in index_order:
        rows = list(np.where(pacmap_clusters == label)[0])
        group_selected.append(rows)
        print(f"Cluster {label}: {len(rows)} samples")
    
    # Save the cluster groups
    with open(os.path.join(results_folder, "cluster_groups.txt"), "w") as f:
        for i, group in enumerate(group_selected):
            f.write(f"Cluster {index_order[i]}: {len(group)} samples\n")
            f.write(f"{group}\n\n")
    
    print("Analysis complete!")