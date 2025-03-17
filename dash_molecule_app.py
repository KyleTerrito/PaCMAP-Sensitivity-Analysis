# dash_molecule_app.py
import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
import base64
from rdkit import Chem
from rdkit.Chem import Draw
import io
import os

def create_dash_app(folder_path="RESULTS/2025-03-09_20-03-29-logcmc", port=8050):
    """
    Create a Dash application for molecular visualization
    
    Parameters:
    folder_path (str): Path to the folder containing data files
    port (int): Port to run the Dash server on
    
    Returns:
    app: Dash application instance
    """
    print(f"Starting Dash app with folder: {folder_path}, port: {port}")
    
    # Load results folder
    folder = folder_path

    # Load all data files
    try:
        print(f"Attempting to load data from: {folder}")
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
                print(f"Found SMILES file at: {location}")
                break
            except FileNotFoundError:
                continue
        
        if molecular_data is None:
            print("Warning: Could not find molecular_descriptors file in common locations. Searching directories...")
            
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
                return None
                
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    print("Successfully loaded all data files")
    
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
            # Basic SMILES parsing
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Just use the default RDKit drawing
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
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

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
        
        # Instructions
        html.Div(style={'marginBottom': '20px', 'textAlign': 'center'}, children=[
            html.Div("Use lasso or box select to choose molecules", 
                    style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            html.Div("Selected molecules will appear below the plot",
                    style={'fontSize': '0.9em', 'color': '#666'})
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
            dragmode='lasso',  # Default to lasso selection
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
    
    print(f"Dash app created successfully, ready to run on port {port}")
    return app

# Only run the server when this script is executed directly
if __name__ == '__main__':
    import sys
    
    # Parse command line arguments if provided
    folder_path = "RESULTS/2025-03-09_20-03-29-logcmc"
    port = 8050
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    app = create_dash_app(folder_path, port)
    if app:
        app.run_server(debug=True, port=port)
    else:
        print("Failed to create Dash app")

# # dash_molecule_app.py
# import dash
# from dash import dcc, html, dash_table
# import plotly.express as px
# import pandas as pd
# from dash.dependencies import Input, Output, State
# import base64
# from rdkit import Chem
# from rdkit.Chem import Draw
# import io
# import os

# def create_dash_app(folder_path="RESULTS/2025-03-09_20-03-29-logcmc", port=8050):
#     # Load results folder
#     folder = folder_path

#     # Load all data files
#     try:
#         original_data = pd.read_csv(f'{folder}/original_data.csv')
#         reduced_data = pd.read_csv(f'{folder}/reduced_data.csv')
#         cluster_labels = pd.read_csv(f'{folder}/cluster_labels.csv')
        
#         # Try to load SMILES file from multiple locations
#         try:
#             molecular_data = pd.read_csv(f'{folder}/original_data_SMILES.csv')
#             print("Found SMILES file in results folder")
#         except FileNotFoundError:
#             try:
#                 molecular_data = pd.read_csv('original_data_SMILES.csv')
#                 print("Found SMILES file in current directory")
#             except FileNotFoundError:
#                 try:
#                     molecular_data = pd.read_csv('DATA/original_data_SMILES.csv')
#                     print("Found SMILES file in DATA folder")
#                 except FileNotFoundError:
#                     print("Warning: Could not find molecular_descriptors file. Searching directories...")
                    
#                     found = False
#                     for root, dirs, files in os.walk('.', topdown=True):
#                         for file in files:
#                             if file == 'original_data_SMILES.csv':
#                                 file_path = os.path.join(root, file)
#                                 print(f"Found SMILES file at: {file_path}")
#                                 molecular_data = pd.read_csv(file_path)
#                                 found = True
#                                 break
#                         if found:
#                             break
                    
#                     if not found:
#                         print("Error: Could not locate the molecular descriptors file.")
#                         print("Please update the path manually in the code.")
#                         molecular_data = pd.DataFrame()
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return None

#     # Ensure reduced_data has an ID column for mapping
#     reduced_data['ID'] = reduced_data.index
#     cluster_labels['ID'] = cluster_labels.index

#     # Create a mapping from ID to SMILES
#     id_to_smiles = {}
#     id_to_name = {}
#     for index, row in original_data.iterrows():
#         name = row['Name']
#         id_val = index + 1  # Convert to 1-based index
#         id_to_name[id_val] = name
        
#         # Find corresponding SMILES
#         mol_row = molecular_data[molecular_data['Name'] == name]
#         if not mol_row.empty and 'SMILES' in mol_row.columns:
#             id_to_smiles[id_val] = mol_row.iloc[0]['SMILES']

#     # Function to generate molecule image from SMILES
#     def smiles_to_image(smiles, width=300, height=200):
#         try:
#             # Basic SMILES parsing
#             mol = Chem.MolFromSmiles(smiles)
#             if mol:
#                 # Just use the default RDKit drawing
#                 img = Draw.MolToImage(mol, size=(width, height))
                
#                 # Convert to base64 for display
#                 buffered = io.BytesIO()
#                 img.save(buffered, format="PNG")
#                 encoded_image = base64.b64encode(buffered.getvalue()).decode()
#                 return f"data:image/png;base64,{encoded_image}"
#         except Exception as e:
#             print(f"Error rendering molecule: {e}")
#             return None
#         return None

#     # Create the Dash App
#     app = dash.Dash(__name__, suppress_callback_exceptions=True)

#     app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'margin': '20px'}, children=[
#         html.H1("Interactive DR & Clustering with Molecule Visualization",
#                 style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
        
#         # Radio Buttons for Color Selection
#         html.Div(style={'marginBottom': '20px', 'textAlign': 'center'}, children=[
#             html.Label("Color By:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
#             dcc.RadioItems(
#                 id='color-option',
#                 options=[
#                     {'label': 'Default', 'value': 'default'},
#                     {'label': 'Cluster Labels', 'value': 'cluster'},
#                     {'label': 'Target Values', 'value': 'target'}
#                 ],
#                 value='default',  # Default selection
#                 inline=True,
#                 style={'display': 'inline-block'}
#             )
#         ]),
        
#         # Scatter Plot
#         dcc.Graph(
#             id='scatter-plot',
#             config={'modeBarButtonsToAdd': ['lasso2d', 'select2d']}
#         ),
        
#         # Molecule visualization container
#         html.Div([
#             html.H3("Selected Molecule Structures", style={'marginTop': '20px', 'marginBottom': '15px'}),
#             html.Div(id='molecule-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
#         ]),
        
#         # Table to display original data points on selection
#         html.Div(id='selected-data-table', style={'marginTop': '20px'}),
        
#         # Display original data when hovering
#         html.Div(id='hover-data', style={'marginTop': '20px'}),
        
#         # Store for precomputed molecule images
#         dcc.Store(id='molecule-images-store')
#     ])

#     @app.callback(
#         Output('molecule-images-store', 'data'),
#         Input('scatter-plot', 'selectedData')
#     )
#     def precompute_molecule_images(selectedData):
#         """Precompute molecule images for selected data points"""
#         if not selectedData or 'points' not in selectedData:
#             return {}
        
#         selected_indices = []
#         for point in selectedData['points']:
#             if 'customdata' in point and point['customdata']:
#                 selected_indices.append(point['customdata'][0])
#             elif 'pointIndex' in point:
#                 selected_indices.append(point['pointIndex'] + 1)
        
#         image_dict = {}
#         for idx in selected_indices:
#             if idx in id_to_smiles:
#                 smiles = id_to_smiles[idx]
#                 img_data = smiles_to_image(smiles)
#                 if img_data:
#                     image_dict[str(idx)] = {
#                         'image': img_data,
#                         'name': id_to_name.get(idx, f"Molecule {idx}")
#                     }
        
#         return image_dict

#     @app.callback(
#         Output('scatter-plot', 'figure'),
#         Input('color-option', 'value')
#     )
#     def update_scatter_plot(color_option):
#         if color_option == 'cluster':
#             # Make sure Label is treated as a categorical variable
#             # First, convert to string to ensure it's treated as discrete
#             reduced_data_copy = reduced_data.copy()
#             reduced_data_copy['Label'] = reduced_data_copy['Label'].astype(str)
            
#             # Create the scatter plot with categorical coloring
#             fig = px.scatter(
#                 reduced_data_copy, 
#                 x='Dim1', y='Dim2',
#                 color='Label',  # Use the Label column for coloring
#                 hover_data={'ID': True},
#                 custom_data=['ID'],
#                 color_discrete_sequence=px.colors.qualitative.Plotly,  # Use a discrete color sequence
#                 category_orders={"Label": sorted(reduced_data_copy['Label'].unique())}  # Sort the labels
#             )
#         elif color_option == 'target':
#             fig = px.scatter(
#                 reduced_data, x='Dim1', y='Dim2',
#                 color=original_data['TARGET'],
#                 color_continuous_scale='Viridis',
#                 hover_data={'ID': True},
#                 custom_data=['ID']
#             )
#             fig.update_coloraxes(colorbar_title="Target Value")
#         else:  # Default
#             fig = px.scatter(
#                 reduced_data, x='Dim1', y='Dim2',
#                 hover_data={'ID': True},
#                 custom_data=['ID']
#             )
#             fig.update_traces(marker=dict(size=8, color='blue'))
        
#         fig.update_traces(
#             selected=dict(marker=dict(color='red', size=10)),
#             unselected=dict(marker=dict(opacity=0.3))
#         )
#         fig.update_layout(
#             uirevision='constant',
#             plot_bgcolor='rgba(240, 240, 240, 0.5)',
#             paper_bgcolor='rgba(240, 240, 240, 0.5)',
#             margin=dict(l=40, r=40, t=40, b=40),
#         )
#         return fig

#     @app.callback(
#         [Output('selected-data-table', 'children'),
#          Output('molecule-container', 'children')],
#         [Input('scatter-plot', 'selectedData'),
#          Input('molecule-images-store', 'data')]
#     )
#     def update_selection(selectedData, molecule_images):
#         if not selectedData or 'points' not in selectedData:
#             return html.Div("Select points to view details"), []
        
#         selected_indices = []
#         for point in selectedData['points']:
#             if 'customdata' in point and point['customdata']:
#                 selected_indices.append(point['customdata'][0])
#             elif 'pointIndex' in point:
#                 selected_indices.append(point['pointIndex'] + 1)
        
#         if not selected_indices:
#             return html.Div("No valid points selected"), []
        
#         # Convert to 0-based indices for DataFrame
#         zero_based_indices = [i - 1 for i in selected_indices]
#         selected_df = original_data.iloc[zero_based_indices]
        
#         # Create molecule visualizations
#         molecule_cards = []
#         for idx in selected_indices:
#             str_idx = str(idx)
#             if molecule_images and str_idx in molecule_images:
#                 mol_data = molecule_images[str_idx]
#                 molecule_cards.append(
#                     html.Div([
#                         html.H4(mol_data['name'], style={'textAlign': 'center', 'marginBottom': '10px', 
#                                                          'height': '40px', 'overflow': 'hidden'}),
#                         html.Img(src=mol_data['image'], style={'maxWidth': '250px', 'margin': 'auto', 'display': 'block'})
#                     ], style={'border': '1px solid #ddd', 'borderRadius': '8px', 
#                              'padding': '15px', 'margin': '10px', 'width': '280px',
#                              'backgroundColor': '#f8f9fa', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
#                 )
        
#         # Create data table
#         table = html.Div([
#             html.H3(f"Selected {len(selected_indices)} Points", style={'marginBottom': '15px'}),
#             dash_table.DataTable(
#                 data=selected_df.to_dict('records'),
#                 columns=[{"name": i, "id": i} for i in selected_df.columns],
#                 style_table={'overflowX': 'auto'},
#                 style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
#                 style_cell={'textAlign': 'left', 'padding': '8px'},
#                 page_size=10
#             )
#         ])
        
#         return table, molecule_cards

#     @app.callback(
#         Output('hover-data', 'children'),
#         Input('scatter-plot', 'hoverData')
#     )
#     def display_hover_data(hoverData):
#         if not hoverData or 'points' not in hoverData or not hoverData['points']:
#             return html.Div("Hover over a point to see details")
        
#         point_info = hoverData['points'][0]
#         if 'customdata' in point_info and point_info['customdata']:
#             point_index = point_info['customdata'][0]
#             zero_based_index = point_index - 1
#         elif 'pointIndex' in point_info:
#             zero_based_index = point_info['pointIndex']
#             point_index = zero_based_index + 1
#         else:
#             return html.Div("Missing point information")
        
#         try:
#             # Get data
#             original_name = original_data.iloc[zero_based_index]['Name']
#             hover_df = original_data.iloc[[zero_based_index]]
            
#             # Get molecule image
#             molecule_img = None
#             if point_index in id_to_smiles:
#                 smiles = id_to_smiles[point_index]
#                 img_data = smiles_to_image(smiles, width=400, height=300)
#                 if img_data:
#                     molecule_img = html.Img(src=img_data, style={'maxWidth': '400px', 'margin': 'auto', 'display': 'block'})
            
#             return html.Div(style={'border': '1px solid #ddd', 'borderRadius': '8px', 'padding': '20px', 
#                                   'backgroundColor': '#f8f9fa', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
#                 html.H3(f"Molecule: {original_name}", style={'borderBottom': '1px solid #eee', 
#                                                             'paddingBottom': '10px', 'marginBottom': '15px'}),
#                 html.Div(style={'display': 'flex', 'flexWrap': 'wrap'}, children=[
#                     # Molecule image
#                     html.Div(molecule_img, style={'flex': '1', 'minWidth': '400px', 'marginRight': '20px', 
#                                                  'marginBottom': '20px', 'textAlign': 'center'}),
                    
#                     # Property highlights
#                     html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
#                         html.H4("Key Properties", style={'marginBottom': '10px'}),
#                         html.Table(style={'width': '100%', 'borderCollapse': 'collapse'}, children=[
#                             html.Tr([
#                                 html.Td("Target Value:", style={'fontWeight': 'bold', 'padding': '8px', 'border': '1px solid #ddd'}),
#                                 html.Td(f"{hover_df.iloc[0]['TARGET']:.3f}", style={'padding': '8px', 'border': '1px solid #ddd'})
#                             ]),
#                             html.Tr([
#                                 html.Td("MW:", style={'fontWeight': 'bold', 'padding': '8px', 'border': '1px solid #ddd'}),
#                                 html.Td(f"{hover_df.iloc[0]['MW']:.1f}", style={'padding': '8px', 'border': '1px solid #ddd'})
#                             ]),
#                             html.Tr([
#                                 html.Td("LogP:", style={'fontWeight': 'bold', 'padding': '8px', 'border': '1px solid #ddd'}),
#                                 html.Td(f"{hover_df.iloc[0]['LogP']:.2f}", style={'padding': '8px', 'border': '1px solid #ddd'})
#                             ]),
#                             html.Tr([
#                                 html.Td("TPSA:", style={'fontWeight': 'bold', 'padding': '8px', 'border': '1px solid #ddd'}),
#                                 html.Td(f"{hover_df.iloc[0]['TPSA']:.2f}", style={'padding': '8px', 'border': '1px solid #ddd'})
#                             ])
#                         ])
#                     ])
#                 ]),
                
#                 # Full data table
#                 html.Div(style={'marginTop': '20px'}, children=[
#                     html.H4("All Properties", style={'marginBottom': '10px'}),
#                     dash_table.DataTable(
#                         data=hover_df.to_dict('records'),
#                         columns=[{"name": i, "id": i} for i in hover_df.columns],
#                         style_table={'overflowX': 'auto'},
#                         style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
#                         style_cell={'textAlign': 'left', 'padding': '8px'}
#                     )
#                 ])
#             ])
#         except Exception as e:
#             return html.Div(f"Error displaying hover data: {str(e)}")
    
#     if __name__ == '__main__':
#         app.run_server(debug=True, port=port)
#     else:
#         return app

# # Run this when the script is executed directly
# if __name__ == '__main__':
#     create_dash_app()