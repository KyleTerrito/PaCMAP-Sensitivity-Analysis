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

# Function to generate molecule image from SMILES with improved rendering
def smiles_to_image(smiles, width=300, height=200):
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            # Try to convert to canonical SMILES first if parsing fails
            canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
            mol = Chem.MolFromSmiles(canonical_smiles)
            
        if mol:
            # Clean up the structure - remove explicit hydrogens
            mol = Chem.RemoveAllHs(mol)
            
            # Compute optimal 2D coordinates with multiple attempts
            # First pass with standard parameters
            AllChem.Compute2DCoords(mol)
            
            # For larger molecules, try a more aggressive optimization
            if mol.GetNumAtoms() > 20:
                AllChem.Compute2DCoords(mol, clearConfs=True, canonOrient=False, 
                                        coordMap={}, boundsMat=None, 
                                        cleanIt=True, forceReplaceCoords=True)
            
            # Drawing options for better visibility
            drawer = Draw.MolDraw2DCairo(width, height)
            drawer.SetFontSize(0.8)  # Slightly smaller font for clarity
            
            # Draw options
            draw_options = drawer.drawOptions()
            draw_options.addStereoAnnotation = True
            draw_options.additionalAtomLabelPadding = 0.15  # More space around atom labels
            draw_options.explicitMethyl = False  # Don't show explicit methyl groups
            draw_options.fixedBondLength = 20.0  # Slightly longer bonds
            draw_options.fixedScale = 0.05  # Better scale
            
            # Draw the molecule with the enhanced options
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            png_data = drawer.GetDrawingText()
            
            # Convert PNG data to base64 for display
            encoded_image = base64.b64encode(png_data).decode()
            return f"data:image/png;base64,{encoded_image}"
        
    except Exception as e:
        print(f"Error rendering molecule: {e}")
        try:
            # Fallback to simpler rendering method
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.RemoveAllHs(mol)
                AllChem.Compute2DCoords(mol)
                img = Draw.MolToImage(mol, size=(width, height))
                
                # Convert to base64 for display
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/png;base64,{encoded_image}"
        except Exception as e:
            print(f"Fallback rendering also failed: {e}")
    
    return None

# Create the Dash App
app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'margin': '20px'}, children=[
    html.H1("Interactive DR & Clustering with Molecule Visualization",
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
    
    # Radio Buttons for Color Selection
    html.Div(style={'marginBottom': '20px', 'textAlign': 'center'}, children=[
        html.Label("Color By:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
        dcc.RadioItems(
            id='color-option',
            options=[
                {'label': 'Default', 'value': 'default'},
                {'label': 'Cluster Labels', 'value': 'cluster'},
                {'label': 'Target Values', 'value': 'target'}
            ],
            value='default',  # Default selection
            inline=True,
            style={'display': 'inline-block'}
        )
    ]),
    
    # Scatter Plot
    dcc.Graph(
        id='scatter-plot',
        config={'modeBarButtonsToAdd': ['lasso2d', 'select2d']}
    ),
    
    # Molecule visualization container
    html.Div([
        html.H3("Selected Molecule Structures", style={'marginTop': '20px', 'marginBottom': '15px'}),
        html.Div(id='molecule-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
    ]),
    
    # Table to display original data points on selection
    html.Div(id='selected-data-table', style={'marginTop': '20px'}),
    
    # Display original data when hovering
    html.Div(id='hover-data', style={'marginTop': '20px'}),
    
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
        # Make sure Label is treated as a categorical variable
        # First, convert to string to ensure it's treated as discrete
        reduced_data_copy = reduced_data.copy()
        reduced_data_copy['Label'] = reduced_data_copy['Label'].astype(str)
        
        # Create the scatter plot with categorical coloring
        fig = px.scatter(
            reduced_data_copy, 
            x='Dim1', y='Dim2',
            color='Label',  # Use the Label column for coloring
            hover_data={'ID': True},
            custom_data=['ID'],
            color_discrete_sequence=px.colors.qualitative.Plotly,  # Use a discrete color sequence
            category_orders={"Label": sorted(reduced_data_copy['Label'].unique())}  # Sort the labels
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
        fig.update_traces(marker=dict(size=8, color='blue'))
    
    fig.update_traces(
        selected=dict(marker=dict(color='red', size=10)),
        unselected=dict(marker=dict(opacity=0.3))
    )
    fig.update_layout(
        uirevision='constant',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='rgba(240, 240, 240, 0.5)',
        margin=dict(l=40, r=40, t=40, b=40),
    )
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
                    html.H4(mol_data['name'], style={'textAlign': 'center', 'marginBottom': '10px', 
                                                     'height': '40px', 'overflow': 'hidden'}),
                    html.Img(src=mol_data['image'], style={'maxWidth': '250px', 'margin': 'auto', 'display': 'block'})
                ], style={'border': '1px solid #ddd', 'borderRadius': '8px', 
                         'padding': '15px', 'margin': '10px', 'width': '280px',
                         'backgroundColor': '#f8f9fa', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            )
    
    # Create data table
    table = html.Div([
        html.H3(f"Selected {len(selected_indices)} Points", style={'marginBottom': '15px'}),
        dash_table.DataTable(
            data=selected_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in selected_df.columns],
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            page_size=10
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
            img_data = smiles_to_image(smiles, width=400, height=300)
            if img_data:
                molecule_img = html.Img(src=img_data, style={'maxWidth': '400px', 'margin': 'auto', 'display': 'block'})
        
        return html.Div(style={'border': '1px solid #ddd', 'borderRadius': '8px', 'padding': '20px', 
                              'backgroundColor': '#f8f9fa', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
            html.H3(f"Molecule: {original_name}", style={'borderBottom': '1px solid #eee', 
                                                        'paddingBottom': '10px', 'marginBottom': '15px'}),
            html.Div(style={'display': 'flex', 'flexWrap': 'wrap'}, children=[
                # Molecule image
                html.Div(molecule_img, style={'flex': '1', 'minWidth': '400px', 'marginRight': '20px', 
                                             'marginBottom': '20px', 'textAlign': 'center'}),
                
                # Property highlights
                html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                    html.H4("Key Properties", style={'marginBottom': '10px'}),
                    html.Table(style={'width': '100%', 'borderCollapse': 'collapse'}, children=[
                        html.Tr([
                            html.Td("Target Value:", style={'fontWeight': 'bold', 'padding': '8px', 'border': '1px solid #ddd'}),
                            html.Td(f"{hover_df.iloc[0]['TARGET']:.3f}", style={'padding': '8px', 'border': '1px solid #ddd'})
                        ]),
                        html.Tr([
                            html.Td("MW:", style={'fontWeight': 'bold', 'padding': '8px', 'border': '1px solid #ddd'}),
                            html.Td(f"{hover_df.iloc[0]['MW']:.1f}", style={'padding': '8px', 'border': '1px solid #ddd'})
                        ]),
                        html.Tr([
                            html.Td("LogP:", style={'fontWeight': 'bold', 'padding': '8px', 'border': '1px solid #ddd'}),
                            html.Td(f"{hover_df.iloc[0]['LogP']:.2f}", style={'padding': '8px', 'border': '1px solid #ddd'})
                        ]),
                        html.Tr([
                            html.Td("TPSA:", style={'fontWeight': 'bold', 'padding': '8px', 'border': '1px solid #ddd'}),
                            html.Td(f"{hover_df.iloc[0]['TPSA']:.2f}", style={'padding': '8px', 'border': '1px solid #ddd'})
                        ])
                    ])
                ])
            ]),
            
            # Full data table
            html.Div(style={'marginTop': '20px'}, children=[
                html.H4("All Properties", style={'marginBottom': '10px'}),
                dash_table.DataTable(
                    data=hover_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in hover_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '8px'}
                )
            ])
        ])
    except Exception as e:
        return html.Div(f"Error displaying hover data: {str(e)}")
    
    return html.Div("Hover over a point to see its original values.")

if __name__ == '__main__':
    app.run_server(debug=True)