import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
import base64
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import io

# Load results folder
folder = r'RESULTS\2025-03-09_20-03-29-logcmc'

# Load all data files
original_data = pd.read_csv(f'{folder}\\original_data.csv')
reduced_data = pd.read_csv(f'{folder}\\reduced_data.csv')
cluster_labels = pd.read_csv(f'{folder}\\cluster_labels.csv')

# Load the file with SMILES strings - try multiple possible locations
try:
    # Try directly in the current directory
    molecular_data = pd.read_csv(r'original_data_SMILES.csv')
    print("Found SMILES file in current directory")
except FileNotFoundError:
    try:
        # Try with DATA folder
        molecular_data = pd.read_csv(r'DATA\original_data_SMILES.csv')
        print("Found SMILES file in DATA folder")
    except FileNotFoundError:
        try:
            # Try with results folder
            molecular_data = pd.read_csv(f'{folder}\\original_data_SMILES.csv')
            print("Found SMILES file in results folder")
        except FileNotFoundError:
            print("Warning: Could not find molecular_descriptors file. Searching directories...")
            
            import os
            
            # Search for the file in the current directory and subdirectories
            found = False
            for root, dirs, files in os.walk('.', topdown=True):
                for file in files:
                    if file == 'original_data_SMILES.csv':
                        file_path = os.path.join(root, file)
                        print(f"Found SMILES file at: {file_path}")
                        molecular_data = pd.read_csv(file_path)
                        found = True
                        break
                if found:
                    break
            
            if not found:
                print("Error: Could not locate the molecular descriptors file.")
                print("Please update the path manually in the code.")
                molecular_data = None

# Ensure reduced_data has an ID column for mapping
reduced_data['ID'] = reduced_data.index
cluster_labels['ID'] = cluster_labels.index

# Create a mapping from ID to SMILES
id_to_smiles = {}
id_to_name = {}
for index, row in original_data.iterrows():
    name = row['Name']
    id_val = index + 1  # Convert to 1-based index
    id_to_name[id_val] = name
    
    # Find corresponding SMILES
    mol_row = molecular_data[molecular_data['Name'] == name]
    if not mol_row.empty and 'SMILES' in mol_row.columns:
        id_to_smiles[id_val] = mol_row.iloc[0]['SMILES']

# Function to generate molecule image from SMILES
def smiles_to_image(smiles, width=300, height=200):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Compute 2D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            mol = Chem.RemoveHs(mol)
            
            # Generate image
            img = Draw.MolToImage(mol, size=(width, height))
            
            # Convert to base64 for display
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{encoded_image}"
    except Exception as e:
        print(f"Error rendering molecule: {e}")
        return None
    return None

# Create the Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive DR & Clustering with Molecule Visualization"),
    
    # Radio Buttons for Color Selection
    dcc.RadioItems(
        id='color-option',
        options=[
            {'label': 'Default', 'value': 'default'},
            {'label': 'Cluster Labels', 'value': 'cluster'},
            {'label': 'Target Values', 'value': 'target'}
        ],
        value='default',  # Default selection
        inline=True
    ),
    
    # Scatter Plot
    dcc.Graph(
        id='scatter-plot',
        config={'modeBarButtonsToAdd': ['lasso2d', 'select2d']}
    ),
    
    # Molecule visualization container
    html.Div([
        html.H3("Selected Molecule Structures"),
        html.Div(id='molecule-container', style={'display': 'flex', 'flexWrap': 'wrap'})
    ]),
    
    # Table to display original data points on selection
    html.Div(id='selected-data-table'),
    
    # Display original data when hovering
    html.Div(id='hover-data'),
    
    # Store for precomputed molecule images
    dcc.Store(id='molecule-images-store')
])

@app.callback(
    Output('molecule-images-store', 'data'),
    Input('scatter-plot', 'selectedData')
)
def precompute_molecule_images(selectedData):
    """Precompute molecule images for selected data points"""
    if not selectedData or 'points' not in selectedData:
        return {}
    
    selected_indices = []
    for point in selectedData['points']:
        if 'customdata' in point and point['customdata']:
            selected_indices.append(point['customdata'][0])
        elif 'pointIndex' in point:
            selected_indices.append(point['pointIndex'] + 1)
    
    image_dict = {}
    for idx in selected_indices:
        if idx in id_to_smiles:
            smiles = id_to_smiles[idx]
            img_data = smiles_to_image(smiles)
            if img_data:
                image_dict[str(idx)] = {
                    'image': img_data,
                    'name': id_to_name.get(idx, f"Molecule {idx}")
                }
    
    return image_dict

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('color-option', 'value')
)
def update_scatter_plot(color_option):
    if color_option == 'cluster':
        fig = px.scatter(
            reduced_data, 
            x='Dim1', y='Dim2',
            color='Label',
            hover_data={'ID': True},
            custom_data=['ID'],
            color_continuous_scale=px.colors.qualitative.G10
        )
    elif color_option == 'target':
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
        fig.update_traces(marker=dict(size=5, color='blue'))
    
    fig.update_traces(
        selected=dict(marker=dict(color='red')),
        unselected=dict(marker=dict(opacity=0.3))
    )
    fig.update_layout(uirevision='constant')
    return fig

@app.callback(
    [Output('selected-data-table', 'children'),
     Output('molecule-container', 'children')],
    [Input('scatter-plot', 'selectedData'),
     Input('molecule-images-store', 'data')]
)
def update_selection(selectedData, molecule_images):
    if not selectedData or 'points' not in selectedData:
        return html.Div("Select points to view details"), []
    
    selected_indices = []
    for point in selectedData['points']:
        if 'customdata' in point and point['customdata']:
            selected_indices.append(point['customdata'][0])
        elif 'pointIndex' in point:
            selected_indices.append(point['pointIndex'] + 1)
    
    if not selected_indices:
        return html.Div("No valid points selected"), []
    
    # Convert to 0-based indices for DataFrame
    zero_based_indices = [i - 1 for i in selected_indices]
    selected_df = original_data.iloc[zero_based_indices]
    
    # Create molecule visualizations
    molecule_cards = []
    for idx in selected_indices:
        str_idx = str(idx)
        if molecule_images and str_idx in molecule_images:
            mol_data = molecule_images[str_idx]
            molecule_cards.append(
                html.Div([
                    html.H4(mol_data['name'], style={'textAlign': 'center'}),
                    html.Img(src=mol_data['image'], style={'maxWidth': '250px'})
                ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 
                         'padding': '10px', 'margin': '5px', 'width': '270px'})
            )
    
    # Create data table
    table = html.Div([
        html.H3(f"Selected {len(selected_indices)} Points"),
        dash_table.DataTable(
            data=selected_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in selected_df.columns],
            style_table={'overflowX': 'auto'},
            page_size=10,
            style_cell={'textAlign': 'left'}
        )
    ])
    
    return table, molecule_cards

@app.callback(
    Output('hover-data', 'children'),
    Input('scatter-plot', 'hoverData')
)
def display_hover_data(hoverData):
    if not hoverData or 'points' not in hoverData or not hoverData['points']:
        return html.Div("Hover over a point to see details")
    
    point_info = hoverData['points'][0]
    if 'customdata' in point_info and point_info['customdata']:
        point_index = point_info['customdata'][0]
        zero_based_index = point_index - 1
    elif 'pointIndex' in point_info:
        zero_based_index = point_info['pointIndex']
        point_index = zero_based_index + 1
    else:
        return html.Div("Missing point information")
    
    try:
        # Get data
        original_name = original_data.iloc[zero_based_index]['Name']
        hover_df = original_data.iloc[[zero_based_index]]
        
        # Get molecule image
        molecule_img = None
        if point_index in id_to_smiles:
            smiles = id_to_smiles[point_index]
            img_data = smiles_to_image(smiles, width=350, height=250)
            if img_data:
                molecule_img = html.Img(src=img_data, style={'maxWidth': '350px'})
        
        return html.Div([
            html.H3(f"Molecule: {original_name}"),
            html.Div([
                html.Div(molecule_img, style={'margin': '10px 0'}),
                dash_table.DataTable(
                    data=hover_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in hover_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'}
                )
            ])
        ])
    except Exception as e:
        return html.Div(f"Error displaying hover data: {str(e)}")

if __name__ == '__main__':
    app.run_server(debug=True)