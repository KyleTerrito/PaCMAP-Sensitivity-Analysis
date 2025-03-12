import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import base64
import io

# Load results folder
folder = r'RESULTS\2025-03-09_20-03-29-logcmc'

# # Load all data files
# original_data = pd.read_csv(f'{folder}\\original_data.csv')
# reduced_data = pd.read_csv(f'{folder}\\reduced_data.csv')
# cluster_labels = pd.read_csv(f'{folder}\\cluster_labels.csv')

# # Load the file with SMILES strings
# molecular_data = pd.read_csv(f'{folder}\\original_data_SMILES.csv')

# Load results folder
folder = r'RESULTS\2025-03-09_20-03-29-logcmc'

# Define file paths
original_data_path = f'{folder}\\original_data.csv'
reduced_data_path = f'{folder}\\reduced_data.csv'
cluster_labels_path = f'{folder}\\cluster_labels.csv'
original_data_smiles_path = f'{folder}\\original_data_SMILES.csv'  # New file with SMILES

# Load all data files
try:
    original_data = pd.read_csv(original_data_path)
    reduced_data = pd.read_csv(reduced_data_path)
    cluster_labels = pd.read_csv(cluster_labels_path)
    print("Successfully loaded original data, reduced data, and cluster labels")
except Exception as e:
    print(f"Error loading data files: {e}")
    raise

# Try to find the molecular data file with SMILES
smiles_file_found = False
molecular_data = None

# First check if original_data_SMILES.csv exists
try:
    molecular_data = pd.read_csv(original_data_smiles_path)
    print(f"Found SMILES data in: {original_data_smiles_path}")
    smiles_file_found = True
except FileNotFoundError:
    print(f"Could not find {original_data_smiles_path}, trying alternative locations")

# List of possible file locations to try if the first attempt failed
possible_paths = [
    r'molecular_descriptors_concat_data_product2_TargetLogcmc.csv',
    r'DATA\molecular_descriptors_concat_data_product2_TargetLogcmc.csv',
    f'{folder}\\molecular_descriptors_concat_data_product2_TargetLogcmc.csv',
]

# Try each path
for path in possible_paths:
    try:
        molecular_data = pd.read_csv(path)
        print(f"Found SMILES file at: {path}")
        smiles_file_found = True
        break
    except FileNotFoundError:
        continue

# If not found in specific locations, search directories
if not smiles_file_found:
    print("Searching directories for SMILES file...")
    for root, dirs, files in os.walk('.', topdown=True):
        for file in files:
            if "molecular_descriptors" in file.lower() and "targetlogcmc" in file.lower() and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    molecular_data = pd.read_csv(file_path)
                    print(f"Found SMILES file at: {file_path}")
                    smiles_file_found = True
                    break
                except:
                    continue
        if smiles_file_found:
            break

if not smiles_file_found:
    print("Warning: Could not find molecular_descriptors file with SMILES.")
    print("The application will run without molecular structure visualization.")
    # Create an empty dataframe as a placeholder
    molecular_data = pd.DataFrame(columns=['Name', 'SMILES'])

# Ensure reduced_data has an ID column for mapping
reduced_data['ID'] = reduced_data.index
cluster_labels['ID'] = cluster_labels.index

# Create the Dash App
app = dash.Dash(__name__)

# Custom CSS for styling
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'margin': '20px'}, children=[
    html.H1("Interactive DR & Clustering with Molecule Data", 
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
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
            value='default',
            inline=True,
            style={'display': 'inline-block'}
        )
    ]),
    
    # Scatter Plot
    dcc.Graph(
        id='scatter-plot',
        config={'modeBarButtonsToAdd': ['lasso2d', 'select2d']}
    ),
    
    # Instructions
    html.Div(style={'marginTop': '10px', 'marginBottom': '20px', 'textAlign': 'center'}, children=[
        html.P("Select points using the lasso or box select tools to view details", 
               style={'color': '#7f8c8d', 'fontStyle': 'italic'})
    ]),
    
    # Two-column layout for molecule info and data table
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap'}, children=[
        # Left column - SMILES
        html.Div(id='molecule-info', style={'flex': '1', 'minWidth': '300px', 'marginRight': '20px'}),
        
        # Right column - Data table
        html.Div(id='selected-data-table', style={'flex': '2', 'minWidth': '500px'})
    ]),
    
    # Hover data at the bottom
    html.Div(id='hover-data', style={'marginTop': '30px'})
])

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
            color_continuous_scale=px.colors.qualitative.G10  # Use a categorical color scale
        )
    elif color_option == 'target':
        fig = px.scatter(
            reduced_data, x='Dim1', y='Dim2',
            color=original_data['TARGET'],  # Use continuous color mapping
            color_continuous_scale='Viridis',  # Choose a color scale
            hover_data={'ID': True},
            custom_data=['ID']
        )
        fig.update_coloraxes(colorbar_title="Target Value")  # Label the color bar
    else:  # Default (Blue)
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
     Output('molecule-info', 'children')],
    Input('scatter-plot', 'selectedData')
)
def update_selection(selectedData):
    if selectedData and 'points' in selectedData:
        selected_indices = []
        for point in selectedData['points']:
            if 'customdata' in point and point['customdata']:
                selected_indices.append(point['customdata'][0])
            elif 'pointIndex' in point:
                selected_indices.append(point['pointIndex'] + 1)  # Adjust index
        
        if not selected_indices:
            return html.Div("No valid point indices found in selection."), html.Div("No molecules selected")
        
        # Convert to 0-based indices for DataFrame access
        zero_based_indices = [i - 1 for i in selected_indices]
        
        # Get original data
        selected_df = original_data.iloc[zero_based_indices]
        
        # Get molecule information
        molecule_info_cards = []
        
        for i, idx in enumerate(zero_based_indices):
            try:
                name = selected_df.iloc[i]['Name']
                
                # Find the SMILES for this molecule
                smiles = "Not found"
                if smiles_file_found:
                    mol_row = molecular_data[molecular_data['Name'] == name]
                    if not mol_row.empty and 'SMILES' in mol_row.columns:
                        smiles = mol_row.iloc[0]['SMILES']
                
                # Create a card for this molecule
                molecule_info_cards.append(
                    html.Div(style={
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'padding': '15px',
                        'marginBottom': '15px',
                        'backgroundColor': '#f8f9fa'
                    }, children=[
                        html.H4(name, style={'borderBottom': '1px solid #eee', 'paddingBottom': '8px'}),
                        html.P("SMILES:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        html.Div(style={
                            'padding': '10px',
                            'border': '1px solid #ddd',
                            'borderRadius': '3px',
                            'backgroundColor': '#fff',
                            'fontFamily': 'monospace',
                            'fontSize': '12px',
                            'overflowX': 'auto'
                        }, children=smiles),
                        html.P(f"Target value: {selected_df.iloc[i]['TARGET']:.2f}", 
                               style={'marginTop': '10px', 'fontWeight': 'bold'})
                    ])
                )
            except Exception as e:
                molecule_info_cards.append(html.Div(f"Error displaying molecule {idx}: {str(e)}"))
        
        # Create molecule info section
        molecule_section = html.Div([
            html.H3(f"Selected Molecules ({len(selected_indices)})", 
                    style={'marginBottom': '15px'}),
            html.Div(molecule_info_cards)
        ])
        
        # Create data table
        table_section = html.Div([
            html.H3("Molecular Properties", style={'marginBottom': '15px'}),
            dash_table.DataTable(
                data=selected_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in selected_df.columns],
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px'
                },
                page_size=10
            )
        ])
        
        return table_section, molecule_section
    
    return html.Div("Select points using the box or lasso tool to view data."), html.Div(
        html.P("Select points to view molecule details", 
               style={'color': '#7f8c8d', 'fontStyle': 'italic', 'textAlign': 'center'})
    )

@app.callback(
    Output('hover-data', 'children'),
    Input('scatter-plot', 'hoverData')
)
def display_hover_data(hoverData):
    if hoverData and 'points' in hoverData and hoverData['points']:
        point_info = hoverData['points'][0]
        # Attempt to get the point index
        if 'customdata' in point_info and point_info['customdata']:
            point_index = point_info['customdata'][0] - 1  # Convert to 0-based index
        elif 'pointIndex' in point_info:
            point_index = point_info['pointIndex']
        else:
            return html.Div("Hovered point data is missing expected information.")
        
        try:
            # Get data for this point
            original_name = original_data.iloc[point_index]['Name']
            hover_df = original_data.iloc[[point_index]]
            
            # Get SMILES for this molecule
            smiles = "SMILES not available"
            if smiles_file_found:
                mol_row = molecular_data[molecular_data['Name'] == original_name]
                if not mol_row.empty and 'SMILES' in mol_row.columns:
                    smiles = mol_row.iloc[0]['SMILES']
            
            return html.Div(style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px'}, children=[
                html.H3(f"Hovered Point: {original_name}", 
                        style={'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                
                html.Div(style={'display': 'flex', 'flexWrap': 'wrap'}, children=[
                    # SMILES info
                    html.Div(style={'flex': '1', 'minWidth': '300px', 'marginRight': '20px'}, children=[
                        html.P("SMILES:", style={'fontWeight': 'bold'}),
                        html.Div(style={
                            'padding': '10px',
                            'border': '1px solid #ddd',
                            'borderRadius': '3px',
                            'backgroundColor': '#f8f9fa',
                            'fontFamily': 'monospace',
                            'fontSize': '12px',
                            'overflowX': 'auto'
                        }, children=smiles)
                    ]),
                    
                    # Key properties
                    html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                        html.P("Key Properties:", style={'fontWeight': 'bold'}),
                        html.Table(style={'width': '100%', 'borderCollapse': 'collapse'}, children=[
                            html.Tr([
                                html.Td("Target Value:", style={'fontWeight': 'bold', 'padding': '5px', 'border': '1px solid #ddd'}),
                                html.Td(f"{hover_df.iloc[0]['TARGET']:.3f}", style={'padding': '5px', 'border': '1px solid #ddd'})
                            ]),
                            html.Tr([
                                html.Td("MW:", style={'fontWeight': 'bold', 'padding': '5px', 'border': '1px solid #ddd'}),
                                html.Td(f"{hover_df.iloc[0]['MW']:.1f}", style={'padding': '5px', 'border': '1px solid #ddd'})
                            ]),
                            html.Tr([
                                html.Td("LogP:", style={'fontWeight': 'bold', 'padding': '5px', 'border': '1px solid #ddd'}),
                                html.Td(f"{hover_df.iloc[0]['LogP']:.2f}", style={'padding': '5px', 'border': '1px solid #ddd'})
                            ]),
                            html.Tr([
                                html.Td("TPSA:", style={'fontWeight': 'bold', 'padding': '5px', 'border': '1px solid #ddd'}),
                                html.Td(f"{hover_df.iloc[0]['TPSA']:.2f}", style={'padding': '5px', 'border': '1px solid #ddd'})
                            ])
                        ])
                    ])
                ]),
                
                # Full data table
                html.Div(style={'marginTop': '15px'}, children=[
                    dash_table.DataTable(
                        data=hover_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in hover_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        style_cell={'textAlign': 'left'}
                    )
                ])
            ])
        except Exception as e:
            return html.Div(f"Error displaying hover data: {str(e)}")
    
    return html.Div("Hover over a point to see its original values.")

if __name__ == '__main__':
    app.run_server(debug=True)

# # Create the Dash App
# app = dash.Dash(__name__)

# # Add external JavaScript to render molecules
# app.index_string = '''
# <!DOCTYPE html>
# <html>
#     <head>
#         {%metas%}
#         <title>{%title%}</title>
#         {%favicon%}
#         {%css%}
#         <!-- RDKit.js for molecule rendering -->
#         <script src="https://unpkg.com/@rdkit/rdkit@2023.9.4/Code/MinimalLib/dist/RDKit_minimal.js"></script>
#     </head>
#     <body>
#         {%app_entry%}
#         <footer>
#             {%config%}
#             {%scripts%}
#             {%renderer%}
#             <script>
#                 // Initialize RDKit
#                 window.onRDKitReady = function() {
#                     console.log('RDKit is ready');
#                 }
#             </script>
#         </footer>
#     </body>
# </html>
# '''

# app.layout = html.Div([
#     html.H1("Interactive DR & Clustering with Molecule Visualization"),
    
#     # Radio Buttons for Color Selection
#     dcc.RadioItems(
#         id='color-option',
#         options=[
#             {'label': 'Default', 'value': 'default'},
#             {'label': 'Cluster Labels', 'value': 'cluster'},
#             {'label': 'Target Values', 'value': 'target'}
#         ],
#         value='default',  # Default selection
#         inline=True
#     ),
    
#     # Scatter Plot
#     dcc.Graph(
#         id='scatter-plot',
#         config={'modeBarButtonsToAdd': ['lasso2d', 'select2d']}
#     ),
    
#     # Molecule visualization container
#     html.Div([
#         html.H3("Selected Molecule Structure"),
#         html.Div(id='molecule-container', style={'display': 'flex', 'flexWrap': 'wrap'})
#     ]),
    
#     # Table to display original data points on selection
#     html.Div(id='selected-data-table'),
    
#     # Display original data when hovering
#     html.Div(id='hover-data'),
    
#     # Hidden div to store SMILES data
#     html.Div(id='smiles-data', style={'display': 'none'})
# ])

# # Generate the JavaScript to render a molecule
# def generate_molecule_js(smiles, div_id, width=250, height=200):
#     return f'''
#     <div id="{div_id}" style="width: {width}px; height: {height}px; display: inline-block;"></div>
#     <script>
#         (function() {{
#             const smiles = "{smiles}";
#             const div_id = "{div_id}";
#             const width = {width};
#             const height = {height};
            
#             if (window.RDKit) {{
#                 const mol = window.RDKit.get_mol(smiles);
#                 if (mol) {{
#                     const canvas = document.getElementById(div_id);
#                     mol.draw_to_canvas_with_highlights(canvas, {{width: width, height: height}});
#                 }}
#             }}
#         }})();
#     </script>
#     '''

# @app.callback(
#     Output('scatter-plot', 'figure'),
#     Input('color-option', 'value')
# )
# def update_scatter_plot(color_option):
#     if color_option == 'cluster':
#         fig = px.scatter(
#             reduced_data, 
#             x='Dim1', y='Dim2',
#             color='Label',
#             hover_data={'ID': True},
#             custom_data=['ID'],
#             color_continuous_scale=px.colors.qualitative.G10  # Use a categorical color scale
#         )
#     elif color_option == 'target':
#         fig = px.scatter(
#             reduced_data, x='Dim1', y='Dim2',
#             color=original_data['TARGET'],  # Use continuous color mapping
#             color_continuous_scale='Viridis',  # Choose a color scale
#             hover_data={'ID': True},
#             custom_data=['ID']
#         )
#         fig.update_coloraxes(colorbar_title="Target Value")  # Label the color bar
#     else:  # Default (Blue)
#         fig = px.scatter(
#             reduced_data, x='Dim1', y='Dim2',
#             hover_data={'ID': True},
#             custom_data=['ID']
#         )
#         fig.update_traces(marker=dict(size=5, color='blue'))
    
#     fig.update_traces(
#         selected=dict(marker=dict(color='red')),
#         unselected=dict(marker=dict(opacity=0.3))
#     )
#     fig.update_layout(uirevision='constant')
#     return fig

# @app.callback(
#     [Output('selected-data-table', 'children'),
#      Output('molecule-container', 'children')],
#     Input('scatter-plot', 'selectedData')
# )
# def update_selection(selectedData):
#     if selectedData and 'points' in selectedData:
#         selected_indices = []
#         for point in selectedData['points']:
#             if 'customdata' in point and point['customdata']:
#                 selected_indices.append(point['customdata'][0])
#             elif 'pointIndex' in point:
#                 selected_indices.append(point['pointIndex'] + 1)  # Adjust index
        
#         if not selected_indices:
#             return html.Div("No valid point indices found in selection."), []
        
#         # Convert to 0-based indices for DataFrame access
#         zero_based_indices = [i - 1 for i in selected_indices]
        
#         # Get original data and corresponding SMILES
#         selected_df = original_data.iloc[zero_based_indices]
        
#         # Create list of molecule visualizations
#         molecule_visualizations = []
#         for i, idx in enumerate(zero_based_indices):
#             try:
#                 name = selected_df.iloc[i]['Name']
#                 # Find the SMILES for this molecule
#                 mol_row = molecular_data[molecular_data['Name'] == name]
                
#                 if not mol_row.empty and 'SMILES' in mol_row.columns:
#                     smiles = mol_row.iloc[0]['SMILES']
#                     mol_div_id = f"mol-{idx}"
#                     molecule_visualizations.append(
#                         html.Div([
#                             html.Div(name, style={'textAlign': 'center', 'fontWeight': 'bold'}),
#                             html.Div(dangerously_allow_html=True, 
#                                     children=generate_molecule_js(smiles, mol_div_id))
#                         ], style={'margin': '10px', 'border': '1px solid #ddd', 'padding': '5px'})
#                     )
#             except Exception as e:
#                 print(f"Error rendering molecule {idx}: {e}")
        
#         # Create data table
#         table = html.Div([
#             html.H3(f"Selected {len(selected_indices)} Points in Original Dataset"),
#             dash_table.DataTable(
#                 data=selected_df.to_dict('records'),
#                 columns=[{"name": i, "id": i} for i in selected_df.columns],
#                 style_table={'overflowX': 'auto'},
#                 page_size=10
#             )
#         ])
        
#         return table, molecule_visualizations
    
#     return html.Div("Select multiple points using the box or lasso tool to view original values."), []

# @app.callback(
#     Output('hover-data', 'children'),
#     Input('scatter-plot', 'hoverData')
# )
# def display_hover_data(hoverData):
#     if hoverData and 'points' in hoverData and hoverData['points']:
#         point_info = hoverData['points'][0]
#         # Attempt to get the point index
#         if 'customdata' in point_info and point_info['customdata']:
#             point_index = point_info['customdata'][0] - 1  # Convert to 0-based index
#         elif 'pointIndex' in point_info:
#             point_index = point_info['pointIndex']
#         else:
#             return html.Div("Hovered point data is missing expected information.")
        
#         try:
#             original_name = original_data.iloc[point_index]['Name']
#             hover_df = original_data.iloc[[point_index]]
            
#             # Get SMILES for this molecule
#             mol_row = molecular_data[molecular_data['Name'] == original_name]
            
#             molecule_vis = []
#             if not mol_row.empty and 'SMILES' in mol_row.columns:
#                 smiles = mol_row.iloc[0]['SMILES']
#                 mol_div_id = f"hover-mol-{point_index}"
#                 molecule_vis = html.Div(
#                     dangerously_allow_html=True,
#                     children=generate_molecule_js(smiles, mol_div_id, width=300, height=250)
#                 )
            
#             return html.Div([
#                 html.H3(f"Hovered Point: {original_name}"),
#                 molecule_vis,
#                 dash_table.DataTable(
#                     data=hover_df.to_dict('records'),
#                     columns=[{"name": i, "id": i} for i in hover_df.columns],
#                     style_table={'overflowX': 'auto'}
#                 )
#             ])
#         except Exception as e:
#             return html.Div(f"Error displaying hover data: {str(e)}")
    
#     return html.Div("Hover over a point to see its original values.")

# if __name__ == '__main__':
#     app.run_server(debug=True)