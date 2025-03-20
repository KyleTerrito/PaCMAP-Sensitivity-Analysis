# streamlit_molecule_app.py
import streamlit as st
import plotly.express as px
import pandas as pd
import base64
from rdkit import Chem
from rdkit.Chem import Draw
import io
import os
import numpy as np
from PIL import Image

def run_streamlit_app(folder_path="RESULTS/2025-03-09_20-03-29-logcmc"):
    """
    Run a Streamlit application for molecular visualization
    
    Parameters:
    folder_path (str): Path to the folder containing data files
    """
    st.set_page_config(layout="wide")
    
    # Set up app title
    st.title("Interactive DR & Clustering with Molecule Visualization")
    
    # Load results folder
    folder = folder_path

    # Load all data files
    try:
        st.info(f"Loading data from: {folder}")
        original_data = pd.read_csv(f'{folder}/original_data.csv')
        reduced_data = pd.read_csv(f'{folder}/reduced_data.csv')
        cluster_labels = pd.read_csv(f'{folder}/cluster_labels.csv')
        
        # Try to load SMILES file from multiple locations
        molecular_data = None
        smiles_locations = [
            f'{folder}/original_data_SMILES.csv',
            'original_data_SMILES.csv',
            'DATA/original_data_SMILES.csv'
        ]
        
        for location in smiles_locations:
            try:
                molecular_data = pd.read_csv(location)
                st.info(f"Found SMILES file at: {location}")
                break
            except FileNotFoundError:
                continue
        
        if molecular_data is None:
            st.warning("Could not find molecular_descriptors file in common locations. Searching directories...")
            
            found = False
            for root, dirs, files in os.walk('.', topdown=True):
                for file in files:
                    if file == 'original_data_SMILES.csv':
                        file_path = os.path.join(root, file)
                        st.info(f"Found SMILES file at: {file_path}")
                        molecular_data = pd.read_csv(file_path)
                        found = True
                        break
                if found:
                    break
            
            if not found:
                st.error("Could not locate the molecular descriptors file.")
                st.error("Please update the path manually in the code.")
                return None
                
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    st.success("Successfully loaded all data files")
    
    # Ensure reduced_data has an ID column for mapping
    reduced_data['ID'] = reduced_data.index
    cluster_labels['ID'] = cluster_labels.index

    # Create a mapping from ID to SMILES
    id_to_smiles = {}
    id_to_name = {}
    for index, row in original_data.iterrows():
        name = row['Name']
        id_val = index + 1 # Convert to 1-based index
        id_to_name[id_val] = name
        
        # Find corresponding SMILES
        mol_row = molecular_data[molecular_data['Name'] == name]
        if not mol_row.empty and 'SMILES' in mol_row.columns:
            id_to_smiles[id_val] = mol_row.iloc[0]['SMILES']

    # Function to generate molecule image from SMILES
    @st.cache_data
    def smiles_to_image(smiles, width=300, height=200):
        try:
            # Basic SMILES parsing
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Just use the default RDKit drawing
                img = Draw.MolToImage(mol, size=(width, height))
                return img
        except Exception as e:
            st.error(f"Error rendering molecule: {e}")
            return None
        return None
    
    # Convert PIL Image to base64 for display
    def get_image_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    # Find the cluster column (it might be named differently than 'Label')
    if 'Label' in cluster_labels.columns:
        cluster_column = 'Label'
    else:
        # Look for any column that might contain cluster information
        potential_columns = [col for col in cluster_labels.columns if col != 'ID']
        if len(potential_columns) > 0:
            cluster_column = potential_columns[0]  # Use the first available column
            st.info(f"Using '{cluster_column}' as cluster column instead of 'Label'")
        else:
            # Fallback: create a dummy column if no suitable column is found
            cluster_column = 'Cluster'
            cluster_labels[cluster_column] = 0
            st.warning("No cluster column found, using dummy values")
    
    # Add cluster information to molecules
    merged_data = reduced_data.copy()
    merged_data['Cluster'] = cluster_labels[cluster_column].astype(float)
    
    # Get unique cluster values for the dropdown
    unique_clusters = sorted(merged_data['Cluster'].unique())
    
    # Create sidebar with controls
    st.sidebar.header("Controls")
    
    # Radio Buttons for Color Selection
    color_option = st.sidebar.radio(
        "Color By:",
        ["Default", "Cluster Labels", "Target Values"],
        index=0
    )
    
    # Add cluster selection dropdown in sidebar
    st.sidebar.header("Cluster Selection")
    selected_cluster = st.sidebar.selectbox(
        "Select Cluster to View Molecules:",
        ["All Clusters"] + [f"Cluster {cluster}" for cluster in unique_clusters]
    )
    
    # Instructions
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("1. Use the dropdown above to select a specific cluster")
    st.sidebar.markdown("2. All molecules in the selected cluster will be displayed below")
    st.sidebar.markdown("3. You can also manually select molecules by index")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    # Store selected indices based on the selection method
    selected_indices = []
    
    with col1:
        # Create the scatter plot based on color option
        if color_option == "Cluster Labels":            
            # Create the scatter plot with continuous coloring for clusters
            fig = px.scatter(
                merged_data, 
                x='Dim1', y='Dim2',
                color='Cluster',  # Use the Cluster column for coloring
                hover_data={'ID': True, 'Cluster': True},
                custom_data=['ID'],
                color_continuous_scale='Viridis',  # Use a continuous color scale
            )
            fig.update_coloraxes(colorbar_title="Cluster")
            
        elif color_option == "Target Values":
            fig = px.scatter(
                reduced_data, x='Dim1', y='Dim2',
                color=original_data['TARGET'],
                color_continuous_scale='Viridis',
                hover_data={'ID': True},
                custom_data=['ID']
            )
            fig.update_coloraxes(colorbar_title="Target Value")
            
        else:  # Default
            fig = px.scatter(
                reduced_data, x='Dim1', y='Dim2',
                hover_data={'ID': True},
                custom_data=['ID']
            )
            fig.update_traces(marker=dict(size=8, color='blue'))
        
        # If a specific cluster is selected, highlight those points
        if selected_cluster != "All Clusters":
            cluster_num = float(selected_cluster.replace("Cluster ", ""))
            
            # Get indices of molecules in the selected cluster
            cluster_indices = merged_data[merged_data['Cluster'] == cluster_num]['ID'].tolist()
            
            # Add 1 to convert to 1-based indices
            cluster_indices = [int(idx + 1) for idx in cluster_indices]
            
            # Filter the data for the selected cluster to highlight
            cluster_data = merged_data[merged_data['Cluster'] == cluster_num]
            
            # Add highlighted points for the selected cluster
            fig.add_trace(
                px.scatter(
                    cluster_data, x='Dim1', y='Dim2',
                    custom_data=['ID']
                ).data[0].update(
                    marker=dict(
                        size=12,
                        color='red',
                        line=dict(width=2, color='black')
                    ),
                    name=f"Cluster {cluster_num}"
                )
            )
            
            # Use these indices as the selected indices
            selected_indices = cluster_indices
        
        fig.update_layout(
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            paper_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(l=40, r=40, t=40, b=40),
            # Set dimensions
            width=800,
            height=500,
            # Force equal scaling to maintain proper data representation
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                constrain="domain"
            ),
            yaxis=dict(
                constrain="domain"
            )
        )
        
        # Display the scatter plot
        scatter_chart = st.plotly_chart(fig, use_container_width=True)
        
        # Add manual selection options
        st.markdown("### Manual Selection")
        st.markdown("Enter comma-separated indices to select specific molecules:")
        
        # Get indices of points to select
        indices_input = st.text_input("Enter indices (e.g., 1,5,10):", "")
        
        if indices_input:
            try:
                manual_indices = [int(idx.strip()) for idx in indices_input.split(",") if idx.strip()]
                
                # Override the cluster selection with manual selection
                if manual_indices:
                    selected_indices = manual_indices
                    st.success(f"Selected {len(selected_indices)} points manually")
            except ValueError:
                st.error("Please enter valid integer indices separated by commas")
        
        # Add a button for random selection
        if st.button("Select 5 Random Points"):
            max_index = len(original_data)
            random_indices = sorted(np.random.choice(range(1, max_index+1), size=min(5, max_index), replace=False).tolist())
            
            # Override the cluster selection with random selection
            selected_indices = random_indices
            st.success(f"Randomly selected points: {', '.join(map(str, selected_indices))}")
    
    # Display hover info in sidebar
    with col2:
        st.markdown("### Molecule Details")
        hover_index = st.number_input("Enter molecule index to view details:", min_value=1, max_value=len(original_data), value=1)
        
        zero_based_index = hover_index - 1
        try:
            # Get data
            original_name = original_data.iloc[zero_based_index]['Name']
            hover_df = original_data.iloc[[zero_based_index]]
            
            # Get cluster information
            if hover_index - 1 < len(merged_data):
                cluster_value = merged_data.iloc[hover_index - 1]['Cluster']
                cluster_info = f"Cluster: {cluster_value}"
            else:
                cluster_info = "Cluster: Unknown"
            
            st.markdown(f"**Molecule: {original_name}**")
            st.markdown(f"**{cluster_info}**")
            
            # Display molecule image
            if hover_index in id_to_smiles:
                smiles = id_to_smiles[hover_index]
                img = smiles_to_image(smiles, width=300, height=200)
                if img:
                    st.image(img, caption=original_name, use_container_width=True)
            
            # Display key properties
            st.markdown("#### Key Properties")
            key_props = pd.DataFrame({
                "Property": ["Target Value", "MW", "LogP", "TPSA"],
                "Value": [
                    f"{hover_df.iloc[0]['TARGET']:.3f}",
                    f"{hover_df.iloc[0]['MW']:.1f}",
                    f"{hover_df.iloc[0]['LogP']:.2f}",
                    f"{hover_df.iloc[0]['TPSA']:.2f}"
                ]
            })
            st.table(key_props)
            
            # Option to show all properties
            if st.checkbox("Show All Properties"):
                st.dataframe(hover_df)
                
        except Exception as e:
            st.error(f"Error displaying molecule details: {str(e)}")
    
    # Display selected molecules
    if selected_indices:
        st.markdown("## Selected Molecule Structures")
        
        if selected_cluster != "All Clusters" and not indices_input and not st.session_state.get('random_selected', False):
            st.markdown(f"### Displaying all molecules in {selected_cluster}")
        
        # Convert to 0-based indices for DataFrame
        valid_indices = [i for i in selected_indices if 1 <= i <= len(original_data)]
        zero_based_indices = [i - 1 for i in valid_indices]
        
        if not zero_based_indices:
            st.warning("No valid molecules selected.")
        else:
            selected_df = original_data.iloc[zero_based_indices]
            
            # Add cluster information to the selected data
            if 'Cluster' not in selected_df.columns:
                cluster_info = []
                for idx in valid_indices:
                    zero_idx = idx - 1
                    if 0 <= zero_idx < len(merged_data):
                        cluster_info.append(merged_data.iloc[zero_idx]['Cluster'])
                    else:
                        cluster_info.append(None)
                selected_df['Cluster'] = cluster_info
            
            # Create molecule visualization grid
            st.markdown(f"### Displaying {len(valid_indices)} selected molecules")
            
            # Support pagination for cluster view when many molecules are selected
            molecules_per_page = 9
            
            if len(valid_indices) > molecules_per_page:
                # Calculate total number of pages
                total_pages = (len(valid_indices) + molecules_per_page - 1) // molecules_per_page
                
                # Add page selection
                page_number = st.selectbox(
                    f"Page (showing {molecules_per_page} molecules per page):",
                    range(1, total_pages + 1),
                    index=0
                )
                
                # Calculate start and end indices for the current page
                start_idx = (page_number - 1) * molecules_per_page
                end_idx = min(start_idx + molecules_per_page, len(valid_indices))
                
                # Display only molecules for the current page
                page_indices = valid_indices[start_idx:end_idx]
                st.markdown(f"**Showing molecules {start_idx+1}-{end_idx} of {len(valid_indices)}**")
            else:
                page_indices = valid_indices
            
            # Create columns for the grid
            cols = st.columns(3)  # Create 3 columns for the grid
            
            for i, idx in enumerate(page_indices):
                col_idx = i % 3  # Determine which column to place this molecule
                
                if idx in id_to_smiles:
                    with cols[col_idx]:
                        mol_name = id_to_name.get(idx, f"Molecule {idx}")
                        st.markdown(f"**{mol_name}**")
                        
                        # Display molecule image
                        smiles = id_to_smiles[idx]
                        img = smiles_to_image(smiles)
                        if img:
                            st.image(img, use_container_width=True)
                        
                        # Show key properties
                        zero_idx = idx - 1
                        if 0 <= zero_idx < len(original_data):
                            target_value = original_data.iloc[zero_idx]['TARGET']
                            mw_value = original_data.iloc[zero_idx]['MW']
                            cluster_value = merged_data.iloc[zero_idx]['Cluster']
                            
                            st.markdown(f"**Target:** {target_value:.3f}")
                            st.markdown(f"**MW:** {mw_value:.1f}")
                            st.markdown(f"**Cluster:** {cluster_value}")
            
            # Display data table for selected molecules
            with st.expander("View Selected Data Table", expanded=False):
                # Select key columns to display
                columns_to_show = ['Name', 'TARGET', 'MW', 'LogP', 'TPSA', 'Cluster']
                display_df = selected_df[columns_to_show].copy()
                
                # Display summary statistics
                st.markdown("### Summary Statistics for Selected Molecules")
                summary_stats = display_df.describe()
                st.dataframe(summary_stats)
                
                # Display individual data points
                st.markdown("### Individual Data Points")
                st.dataframe(display_df)
                
                # Add download button for the selected data
                csv = selected_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="selected_molecules.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No molecules selected. Select a cluster from the dropdown or use manual selection.")

# Run the app
if __name__ == '__main__':
    import sys
    
    # Parse command line arguments if provided
    folder_path = "RESULTS/2025-03-09_20-03-29-logcmc"
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    
    run_streamlit_app(folder_path)