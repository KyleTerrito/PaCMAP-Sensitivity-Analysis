import dash
from dash import dcc, html, Input, Output, State, callback, ALL, MATCH
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io
import json
import warnings
import logging
import traceback
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors, AllChem, Fragments
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dash_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Filter deprecation warnings related to date parsing
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message="Parsing dates involving a day of month without a year specified is ambiguious")

# Import PaCMAP with error handling
try:
    import pacmap
    has_pacmap = True
    logger.info("PaCMAP successfully imported")
except ImportError:
    has_pacmap = False
    logger.warning("PaCMAP not available - install with 'pip install pacmap'")

# Initialize the Dash app with Bootstrap theme and error handling
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server  # Needed for deployment

# Global error handling function
def log_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return html.Div([
                html.H5("An error occurred", className="text-danger"),
                html.P(error_msg),
                html.P("Check the console or log files for more details.")
            ])
    return wrapper

# Define the navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Data Preprocessing", href="/preprocessing")),
        dbc.NavItem(dbc.NavLink("Analytics", href="/analytics")),
    ],
    brand="SMILES Data Processor",
    brand_href="/",
    color="primary",
    dark=True,
)

# Define the layout for each page

# Home page
home_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Welcome to SMILES Data Processor"), width=12, className="mb-4 mt-4")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Preprocessing"),
                dbc.CardBody([
                    html.P("Upload data and prepare it for analysis."),
                    dbc.Button("Go to Data Preprocessing", href="/preprocessing", color="primary")
                ])
            ], className="mb-4")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analytics"),
                dbc.CardBody([
                    html.P("Analyze and visualize prepared data."),
                    dbc.Button("Go to Analytics", href="/analytics", color="primary")
                ])
            ], className="mb-4")
        ], width=6)
    ])
])

# Data Preprocessing layout
preprocessing_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Data Preprocessing"), width=12, className="mb-4 mt-4")
    ]),
    dbc.Tabs([
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("Upload your file"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-output'),
                    html.Div(id='dataframe-preview'),
                    html.Div(id='dataset-info')
                ])
            ]),
            dbc.Card([
                dbc.CardHeader("Instructions"),
                dbc.CardBody([
                    html.P([
                        "1. Upload a file containing SMILES data",
                        html.Br(),
                        "2. The file can be one of the following formats:",
                        html.Ul([
                            html.Li("Excel file (.xlsx, .xls) with a column for SMILES"),
                            html.Li("CSV file with a column for SMILES"),
                            html.Li("Text file with one SMILES string per line")
                        ]),
                        "3. The app will display the dataset",
                        html.Br(),
                        "4. After uploading, proceed to Column Selection"
                    ])
                ])
            ], className="mt-4")
        ], label="Data Upload"),
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("Select columns and calculate descriptors"),
                dbc.CardBody([
                    html.Div(id='column-selection-container', children=[
                        html.P("Please upload your data first in the Data Upload tab.")
                    ]),
                    html.Div(id='smiles-column-selection'),
                    html.Div(id='processing-button-container'),
                    html.Div(id='processing-output')
                ])
            ])
        ], label="Column Selection", id="column-selection-tab")
    ])
])

# Analytics layout
analytics_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Analytics"), width=12, className="mb-4 mt-4")
    ]),
    dbc.Tabs([
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("Data Viewer"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Upload(
                                id='analytics-upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            html.Div(id='analytics-upload-output'),
                            html.Div(id='analytics-dataframe-preview'),
                            html.Div(id='analytics-dataset-info')
                        ], label="Upload New Data"),
                        dbc.Tab([
                            html.Div(id='preprocessed-data-container', children=[
                                html.P("Check if preprocessed data is available."),
                                dbc.Button("Check for Preprocessed Data", id="check-preprocessed-data", color="primary"),
                                html.Div(id="preprocessed-data-info")
                            ])
                        ], label="Use Preprocessed Data")
                    ]),
                    html.Div(id='data-exploration-container')
                ])
            ])
        ], label="Data Viewer"),
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("Dimensionality Reduction"),
                dbc.CardBody([
                    html.Div(id='dim-reduction-container')
                ])
            ])
        ], label="Dimensionality Reduction"),
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("Clustering"),
                dbc.CardBody([
                    html.P("Clustering functionality will be implemented here.")
                ])
            ])
        ], label="Clustering"),
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("Visualization"),
                dbc.CardBody([
                    html.P("Visualization functionality will be implemented here.")
                ])
            ])
        ], label="Visualization")
    ])
])

# Layout for the entire app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

# Callback to handle page navigation
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/preprocessing':
        return preprocessing_layout
    elif pathname == '/analytics':
        return analytics_layout
    else:
        return home_layout

# Callback for file upload in Data Preprocessing
@app.callback(
    [Output('upload-output', 'children'),
     Output('dataframe-preview', 'children'),
     Output('dataset-info', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
@log_exception
def update_output(contents, filename):
    if contents is None:
        raise PreventUpdate
    
    # Parse uploaded file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Determine file type
    if filename.endswith('.csv'):
        # Try multiple encodings and delimiters for CSV
        try:
            # First try UTF-8
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            success_msg = html.Div([
                html.H5("File successfully loaded as CSV (UTF-8)", className="text-success")
            ])
        except UnicodeDecodeError:
            try:
                # Try Latin-1
                df = pd.read_csv(io.StringIO(decoded.decode('latin-1')))
                success_msg = html.Div([
                    html.H5("File successfully loaded as CSV (Latin-1)", className="text-success")
                ])
            except:
                try:
                    # Try Windows-1252
                    df = pd.read_csv(io.StringIO(decoded.decode('cp1252')))
                    success_msg = html.Div([
                        html.H5("File successfully loaded as CSV (Windows-1252)", className="text-success")
                    ])
                except Exception as e:
                    return html.Div([
                        html.H5("Error processing CSV file", className="text-danger"),
                        html.P(f"Error: {str(e)}"),
                        html.P("Try converting your CSV to UTF-8 encoding before uploading.")
                    ]), html.Div(), html.Div()
        except Exception as e:
            # Handle other CSV parsing errors
            return html.Div([
                html.H5("Error processing CSV file", className="text-danger"),
                html.P(f"Error: {str(e)}"),
                html.P("Check CSV format, delimiters, or try an Excel file instead.")
            ]), html.Div(), html.Div()
    elif filename.endswith(('.xls', '.xlsx')):
        # Excel file
        df = pd.read_excel(io.BytesIO(decoded))
        success_msg = html.Div([
            html.H5(f"File successfully loaded as Excel ({filename.split('.')[-1]})", className="text-success")
        ])
    else:
        # Assume text file with one SMILES per line
        try:
            content = decoded.decode('utf-8')
            lines = content.strip().split('\n')
            df = pd.DataFrame(lines, columns=['SMILES'])
            success_msg = html.Div([
                html.H5("File successfully loaded as text file", className="text-success")
            ])
        except UnicodeDecodeError:
            try:
                content = decoded.decode('latin-1')
                lines = content.strip().split('\n')
                df = pd.DataFrame(lines, columns=['SMILES'])
                success_msg = html.Div([
                    html.H5("File successfully loaded as text file (Latin-1)", className="text-success")
                ])
            except Exception as e:
                return html.Div([
                    html.H5("Error processing text file", className="text-danger"),
                    html.P(f"Error: {str(e)}")
                ]), html.Div(), html.Div()
    
    # Store the dataframe in app instance
    app.dataframe = df
    
    # Create dataframe preview
    preview = html.Div([
        html.H5("Dataset"),
        dbc.Table.from_dataframe(df.head(10), striped=True, bordered=True, hover=True)
    ])
    
    # Create dataset info
    info = html.Div([
        html.H5("Dataset Information"),
        html.P(f"Number of rows: {len(df)}"),
        html.P(f"Number of columns: {len(df.columns)}"),
        html.P(f"Column names: {', '.join(df.columns)}")
    ])
    
    return success_msg, preview, info

# Callback to enable Column Selection tab after data upload
@app.callback(
    Output('column-selection-container', 'children'),
    Input('upload-output', 'children')
)
@log_exception
def update_column_selection(upload_output):
    if upload_output is None:
        return html.P("Please upload your data first in the Data Upload tab.")
    
    if not hasattr(app, 'dataframe'):
        return html.P("Please upload your data first in the Data Upload tab.")
    
    df = app.dataframe
    
    return html.Div([
        html.H5("Select columns to keep"),
        dcc.Checklist(
            id='column-selection-checklist',
            options=[{'label': col, 'value': col} for col in df.columns],
            value=list(df.columns),
            inline=False
        )
    ])

# Callback to add SMILES column selection
@app.callback(
    Output('smiles-column-selection', 'children'),
    Input('column-selection-checklist', 'value')
)
@log_exception
def update_smiles_selection(selected_columns):
    if not selected_columns:
        return html.Div()
    
    return html.Div([
        html.H5("Select the column containing SMILES strings:", className="mt-3"),
        dcc.Dropdown(
            id='smiles-column-dropdown',
            options=[{'label': col, 'value': col} for col in selected_columns],
            value=selected_columns[0]
        )
    ])

# Callback to add processing button
@app.callback(
    Output('processing-button-container', 'children'),
    Input('smiles-column-dropdown', 'value')
)
@log_exception
def update_processing_button(smiles_column):
    if not smiles_column:
        return html.Div()
    
    return html.Div([
        dbc.Button(
            "Process SMILES and Calculate Descriptors",
            id="process-smiles-button",
            color="primary",
            className="mt-3"
        )
    ])

# Function to calculate descriptors with improved error handling
def calculate_descriptors(smiles):
    """Calculate molecular descriptors for a SMILES string"""
    try:
        # First validate SMILES structure
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"Invalid SMILES structure: {smiles[:50]}..."
        
        # Check for common structural issues
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return None, f"Structure sanitization failed: {str(e)}"
        
        # Basic descriptors
        descriptors = {
            'MW': Descriptors.ExactMolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBA': rdMolDescriptors.CalcNumHBA(mol),
            'HBD': rdMolDescriptors.CalcNumHBD(mol),
            'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'NumAromRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
            'NumRings': rdMolDescriptors.CalcNumRings(mol)
        }
        
        return descriptors, None
    
    except Exception as e:
        error_msg = f"Error calculating descriptors: {str(e)}"
        logger.error(f"SMILES error: {error_msg} for SMILES: {smiles[:50]}...")
        return None, error_msg

# Callback to process SMILES and calculate descriptors
@app.callback(
    Output('processing-output', 'children'),
    Input('process-smiles-button', 'n_clicks'),
    State('column-selection-checklist', 'value'),
    State('smiles-column-dropdown', 'value')
)
@log_exception
def process_smiles(n_clicks, selected_columns, smiles_column):
    if n_clicks is None or not selected_columns or not smiles_column:
        return html.Div()
    
    if not hasattr(app, 'dataframe'):
        return html.Div([
            html.H5("No data available", className="text-danger"),
            html.P("Please upload data first.")
        ])
        
    df = app.dataframe.copy()
    filtered_df = df[selected_columns].copy()
    
    # Create progress indicators
    progress_indicators = html.Div([
        html.H5("Validating SMILES and Calculating Descriptors"),
        dbc.Progress(id="smiles-progress", value=0, striped=True, animated=True)
    ])
    
    # Process SMILES strings and calculate descriptors
    filtered_df['SMILES_Valid'] = False
    
    # Lists to store valid and invalid SMILES
    invalid_smiles = []
    valid_smiles = []
    
    # Dictionary to store descriptor results
    descriptor_results = {}
    
    # Process in chunks to keep the app responsive
    chunk_size = 100
    total_rows = len(filtered_df)
    
    for i in range(0, total_rows, chunk_size):
        chunk = filtered_df.iloc[i:min(i+chunk_size, total_rows)]
        
        for idx, row in chunk.iterrows():
            smiles = row[smiles_column]
            
            # Skip empty strings
            if pd.isna(smiles) or smiles == '':
                invalid_smiles.append((idx, smiles, "Empty SMILES"))
                continue
            
            # Calculate descriptors with improved error handling
            descriptors, error_msg = calculate_descriptors(smiles)
            
            if descriptors is None:
                # Invalid SMILES
                invalid_smiles.append((idx, smiles, error_msg or "Invalid SMILES structure"))
            else:
                # Valid SMILES
                filtered_df.at[idx, 'SMILES_Valid'] = True
                valid_smiles.append((idx, smiles))
                
                # Store descriptors
                for desc_name, desc_value in descriptors.items():
                    if desc_name not in descriptor_results:
                        descriptor_results[desc_name] = {}
                    descriptor_results[desc_name][idx] = desc_value
    
    # Create a new dataframe with descriptors
    for desc_name, desc_values in descriptor_results.items():
        filtered_df[desc_name] = pd.Series(desc_values)
    
    # Store processed data
    app.processed_data = filtered_df
    
    # Create results display
    results = html.Div([
        html.H5("SMILES Validation Results"),
        html.P(f"Valid SMILES: {len(valid_smiles)} of {len(filtered_df)}"),
        html.P(f"Invalid SMILES: {len(invalid_smiles)} of {len(filtered_df)}"),
        
        dbc.Tabs([
            dbc.Tab([
                html.Div([html.P(f"Row {idx+1}: {smiles}")] for idx, smiles in valid_smiles[:20]) 
                if valid_smiles else html.P("No valid SMILES found.")
            ], label="Valid SMILES"),
            dbc.Tab([
                html.Div([html.P(f"Row {idx+1}: {smiles[:50]}... - Error: {reason}")] 
                         for idx, smiles, reason in invalid_smiles[:20])
                if invalid_smiles else html.P("No invalid SMILES found.")
            ], label="Invalid SMILES")
        ]),
        
        html.H5("Processed Dataset with Descriptors", className="mt-3"),
        dbc.Table.from_dataframe(filtered_df.head(10), striped=True, bordered=True, hover=True),
        
        # Only show descriptor statistics if we have valid molecules
        html.Div([
            html.H5("Descriptor Statistics", className="mt-3"),
            dbc.Table.from_dataframe(
                filtered_df[[col for col in filtered_df.columns 
                           if col not in selected_columns and col != 'SMILES_Valid']].describe().round(2),
                striped=True, bordered=True, hover=True
            )
        ]) if valid_smiles else html.Div(),
        
        dbc.Button("Download Processed Data", id="download-button", color="success", className="mt-3 me-2"),
        dbc.Button("Download Valid Molecules Only", id="download-valid-button", color="info", className="mt-3"),
        
        # Add a debug section for SMILES strings with issues
        html.Div([
            html.H5("SMILES Parsing Issues", className="mt-3 text-warning"),
            html.P("The following SMILES strings have structural issues:"),
            html.Ul([
                html.Li([
                    html.Strong(f"Row {idx+1}: "),
                    html.Span(f"{smiles[:30]}..."),
                    html.Br(),
                    html.Small(f"Error: {reason}")
                ]) for idx, smiles, reason in invalid_smiles[:10]
            ]) if invalid_smiles else html.P("No parsing issues found.")
        ]) if invalid_smiles else html.Div()
    ])
    
    return html.Div([progress_indicators, results])

# Callback for data downloads
@app.callback(
    Output('download-data', 'data'),
    [Input('download-button', 'n_clicks'),
     Input('download-valid-button', 'n_clicks')]
)
@log_exception
def download_data(all_click, valid_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not hasattr(app, 'processed_data'):
        raise PreventUpdate
    
    if button_id == 'download-button':
        # Download all processed data
        return dcc.send_data_frame(app.processed_data.to_csv, "processed_smiles_data.csv", index=False)
    elif button_id == 'download-valid-button':
        # Download only valid molecules
        if 'SMILES_Valid' in app.processed_data.columns:
            valid_df = app.processed_data[app.processed_data['SMILES_Valid'] == True].copy()
            return dcc.send_data_frame(valid_df.to_csv, "valid_molecules_data.csv", index=False)
    
    raise PreventUpdate

# Callback for file upload in Analytics
@app.callback(
    [Output('analytics-upload-output', 'children'),
     Output('analytics-dataframe-preview', 'children'),
     Output('analytics-dataset-info', 'children')],
    Input('analytics-upload-data', 'contents'),
    State('analytics-upload-data', 'filename')
)
@log_exception
def update_analytics_output(contents, filename):
    if contents is None:
        raise PreventUpdate
    
    # Parse uploaded file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Determine file type
    if filename.endswith('.csv'):
        # Try multiple encodings and delimiters for CSV
        try:
            # First try UTF-8
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            success_msg = html.Div([
                html.H5("File successfully loaded as CSV (UTF-8)", className="text-success")
            ])
        except UnicodeDecodeError:
            try:
                # Try Latin-1
                df = pd.read_csv(io.StringIO(decoded.decode('latin-1')))
                success_msg = html.Div([
                    html.H5("File successfully loaded as CSV (Latin-1)", className="text-success")
                ])
            except:
                try:
                    # Try Windows-1252
                    df = pd.read_csv(io.StringIO(decoded.decode('cp1252')))
                    success_msg = html.Div([
                        html.H5("File successfully loaded as CSV (Windows-1252)", className="text-success")
                    ])
                except Exception as e:
                    return html.Div([
                        html.H5("Error processing CSV file", className="text-danger"),
                        html.P(f"Error: {str(e)}"),
                        html.P("Try converting your CSV to UTF-8 encoding before uploading.")
                    ]), html.Div(), html.Div()
        except Exception as e:
            # Handle other CSV parsing errors
            return html.Div([
                html.H5("Error processing CSV file", className="text-danger"),
                html.P(f"Error: {str(e)}"),
                html.P("Check CSV format, delimiters, or try an Excel file instead.")
            ]), html.Div(), html.Div()
    elif filename.endswith(('.xls', '.xlsx')):
        # Excel file
        df = pd.read_excel(io.BytesIO(decoded))
        success_msg = html.Div([
            html.H5(f"File successfully loaded as Excel ({filename.split('.')[-1]})", className="text-success")
        ])
    else:
        # Assume text file with one SMILES per line
        try:
            content = decoded.decode('utf-8')
            lines = content.strip().split('\n')
            df = pd.DataFrame(lines, columns=['SMILES'])
            success_msg = html.Div([
                html.H5("File successfully loaded as text file", className="text-success")
            ])
        except UnicodeDecodeError:
            try:
                content = decoded.decode('latin-1')
                lines = content.strip().split('\n')
                df = pd.DataFrame(lines, columns=['SMILES'])
                success_msg = html.Div([
                    html.H5("File successfully loaded as text file (Latin-1)", className="text-success")
                ])
            except Exception as e:
                return html.Div([
                    html.H5("Error processing text file", className="text-danger"),
                    html.P(f"Error: {str(e)}")
                ]), html.Div(), html.Div()
    
    # Store the dataframe for analytics specifically
    app.analytics_data = df
    
    # Create dataframe preview
    preview = html.Div([
        html.H5("Dataset"),
        dbc.Table.from_dataframe(df.head(10), striped=True, bordered=True, hover=True)
    ])
    
    # Create dataset info
    info = html.Div([
        html.H5("Dataset Information"),
        html.P(f"Number of rows: {len(df)}"),
        html.P(f"Number of columns: {len(df.columns)}"),
        html.P(f"Column names: {', '.join(df.columns)}"),
        html.Div([
            dbc.Button("Use This Data for Analytics", 
                      id="use-analytics-data", 
                      color="primary",
                      className="mt-3")
        ])
    ])
    
    return success_msg, preview, info

# Callback to use uploaded analytics data
@app.callback(
    Output('data-exploration-container', 'children'),
    Input('use-analytics-data', 'n_clicks')
)
@log_exception
def use_analytics_data(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    if hasattr(app, 'analytics_data'):
        df = app.analytics_data
        
        # Create basic data exploration interface
        return html.Div([
            html.H5("Data Exploration", className="mt-3"),
            html.P(f"Exploring dataset with {len(df)} rows and {len(df.columns)} columns."),
            
            html.H5("Basic Statistics", className="mt-3"),
            dbc.Table.from_dataframe(
                df.describe().round(2) if not df.empty else pd.DataFrame(),
                striped=True, bordered=True, hover=True
            ),
            
            html.H5("Data Preview", className="mt-3"),
            dbc.Table.from_dataframe(
                df.head(20),
                striped=True, bordered=True, hover=True
            ),
            
            html.Div([
                dbc.Button("Proceed to Dimensionality Reduction", 
                          id="goto-dim-reduction", 
                          color="primary",
                          className="mt-3")
            ])
        ])
    else:
        return html.Div([
            html.H5("No data available", className="text-warning"),
            html.P("Please upload data and click 'Use This Data for Analytics'.")
        ])

# Callback for using preprocessed data
@app.callback(
    [Output('preprocessed-data-info', 'children')],
    [Input('check-preprocessed-data', 'n_clicks')]
)
@log_exception
def check_preprocessed_data(n_clicks):
    if n_clicks is None:
        return [html.Div()]
    
    if hasattr(app, 'processed_data'):
        df = app.processed_data
        app.analytics_data = df  # Make it available for analytics
        
        return [html.Div([
            html.H5("Preprocessed data is available", className="text-success"),
            html.P(f"Dataset contains {len(df)} rows and {len(df.columns)} columns."),
            dbc.Button("Use This Data for Analytics", 
                      id="use-preprocessed-data", 
                      color="primary",
                      className="mt-3")
        ])]
    else:
        return [html.Div([
            html.H5("No preprocessed data available", className="text-warning"),
            html.P("Please process data in the Data Preprocessing section first.")
        ])]

# Define callback for dimensionality reduction
@app.callback(
    Output('dim-reduction-container', 'children'),
    Input('url', 'pathname')
)
@log_exception
def initialize_dim_reduction(pathname):
    if pathname != '/analytics':
        raise PreventUpdate
    
    # Check if data is available
    if hasattr(app, 'analytics_data') or hasattr(app, 'processed_data'):
        df = getattr(app, 'analytics_data', None) or getattr(app, 'processed_data', None)
        
        # Create the layout for dimensionality reduction
        return html.Div([
            html.H5("Select options for dimensionality reduction"),
            
            # Filter valid molecules if column exists
            dbc.Checkbox(
                id="use-only-valid",
                label="Use only valid molecules",
                value=True,
                className="mt-3"
            ) if 'SMILES_Valid' in df.columns else html.Div(),
            
            html.H5("Select columns to exclude from analysis (non-descriptors):", className="mt-3"),
            dcc.Dropdown(
                id="non-descriptor-cols",
                options=[{'label': col, 'value': col} for col in df.columns],
                value=[col for col in df.columns if col in ['SMILES', 'SMILES_Valid', 'Product', 'Name']],
                multi=True
            ),
            
            html.H5("How to handle missing values:", className="mt-3"),
            dcc.Dropdown(
                id="missing-values-handling",
                options=[
                    {'label': 'Drop columns with missing values', 'value': 'drop'},
                    {'label': 'Fill missing values with mean', 'value': 'mean'},
                    {'label': 'Fill missing values with median', 'value': 'median'},
                    {'label': 'Fill missing values with 0', 'value': 'zero'}
                ],
                value='drop'
            ),
            
            dbc.Checkbox(
                id="standardize-data",
                label="Standardize data (recommended)",
                value=True,
                className="mt-3"
            ),
            
            html.H5("Select dimensionality reduction method:", className="mt-3"),
            dcc.Dropdown(
                id="reduction-method",
                options=[
                    {'label': 'PCA', 'value': 'PCA'},
                    {'label': 't-SNE', 'value': 't-SNE'},
                    {'label': 'UMAP', 'value': 'UMAP'},
                    {'label': 'PaCMAP', 'value': 'PaCMAP'} if has_pacmap else None
                ],
                value='PCA'
            ),
            html.Div(id="pacmap-status", children=[
                html.P("PaCMAP is not available. Install with 'pip install pacmap'", 
                       className="text-warning") if not has_pacmap else None
            ]),
            
            html.Div(id="method-specific-params"),
            
            html.H5("Number of components:", className="mt-3"),
            dcc.Slider(
                id="n-components",
                min=2,
                max=min(10, len([col for col in df.columns if col not in ['SMILES', 'SMILES_Valid']])),
                step=1,
                value=2,
                marks={i: str(i) for i in range(2, 11)}
            ),
            
            html.H5("Color points by:", className="mt-3"),
            dcc.Dropdown(
                id="color-by",
                options=[{'label': 'None', 'value': 'None'}] + 
                        [{'label': col, 'value': col} for col in df.columns],
                value='None'
            ),
            
            dbc.Button(
                "Run Dimensionality Reduction",
                id="run-dim-reduction",
                color="primary",
                className="mt-3"
            ),
            
            html.Div(id="dim-reduction-results")
        ])
    else:
        return html.Div([
            html.H5("No data available for analysis."),
            html.P("Please upload data in the Data Viewer or process data in the Data Preprocessing section first.")
        ])

# Add method-specific parameters
@app.callback(
    Output('method-specific-params', 'children'),
    Input('reduction-method', 'value')
)
@log_exception
def update_method_params(method):
    if method == 't-SNE':
        return html.Div([
            html.H5("Perplexity:", className="mt-3"),
            dcc.Slider(
                id="tsne-perplexity",
                min=5,
                max=50,
                step=1,
                value=30,
                marks={i: str(i) for i in range(5, 51, 5)}
            )
        ])
    elif method == 'UMAP':
        return html.Div([
            html.H5("Number of neighbors:", className="mt-3"),
            dcc.Slider(
                id="umap-n-neighbors",
                min=2,
                max=100,
                step=1,
                value=15,
                marks={i: str(i) for i in [2, 5, 10, 15, 20, 30, 50, 100]}
            ),
            
            html.H5("Minimum distance:", className="mt-3"),
            dcc.Slider(
                id="umap-min-dist",
                min=0,
                max=0.99,
                step=0.01,
                value=0.1,
                marks={i/10: str(i/10) for i in range(0, 10, 1)}
            )
        ])
    elif method == 'PaCMAP' and has_pacmap:
        return html.Div([
            html.H5("Number of neighbors:", className="mt-3"),
            dcc.Slider(
                id="pacmap-n-neighbors",
                min=5,
                max=100,
                step=1,
                value=10,
                marks={i: str(i) for i in [5, 10, 20, 30, 50, 100]}
            ),
            
            html.H5("MN ratio:", className="mt-3"),
            dcc.Slider(
                id="pacmap-mn-ratio",
                min=0.1,
                max=1.0,
                step=0.1,
                value=0.5,
                marks={i/10: str(i/10) for i in range(1, 11)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.H5("FP ratio:", className="mt-3"),
            dcc.Slider(
                id="pacmap-fp-ratio",
                min=0.1,
                max=5.0,
                step=0.1,
                value=3.0,
                marks={i: str(i) for i in range(0, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.P("PaCMAP works best with standardized data and may take some time for larger datasets.",
                  className="text-info mt-3")
        ])
    else:  # PCA
        return html.Div()

# Run dimensionality reduction
@app.callback(
    Output('dim-reduction-results', 'children'),
    Input('run-dim-reduction', 'n_clicks'),
    [State('use-only-valid', 'value'),
     State('non-descriptor-cols', 'value'),
     State('missing-values-handling', 'value'),
     State('standardize-data', 'value'),
     State('reduction-method', 'value'),
     State('n-components', 'value'),
     State('color-by', 'value'),
     State('tsne-perplexity', 'value'),
     State('umap-n-neighbors', 'value'),
     State('umap-min-dist', 'value'),
     State('pacmap-n-neighbors', 'value'),
     State('pacmap-mn-ratio', 'value'),
     State('pacmap-fp-ratio', 'value')]
)
@log_exception
def run_dimensionality_reduction(n_clicks, use_only_valid, non_descriptor_cols, missing_values_handling, 
                               standardize, reduction_method, n_components, color_by, 
                               tsne_perplexity, umap_n_neighbors, umap_min_dist,
                               pacmap_n_neighbors, pacmap_mn_ratio, pacmap_fp_ratio):
    if n_clicks is None:
        raise PreventUpdate
    
    # Get the data
    if hasattr(app, 'analytics_data'):
        df = app.analytics_data
    elif hasattr(app, 'processed_data'):
        df = app.processed_data
    else:
        return html.Div([
            html.H5("No data available for analysis", className="text-danger"),
            html.P("Please upload or process data first.")
        ])
    
    # Filter to only valid molecules if desired and if the column exists
    if use_only_valid and 'SMILES_Valid' in df.columns:
        df_analysis = df[df['SMILES_Valid'] == True].copy()
        if df_analysis.empty:
            return html.Div([
                html.H5("No valid molecules found", className="text-warning"),
                html.P("Please adjust your filter or use all molecules.")
            ])
    else:
        df_analysis = df.copy()
    
    # Get descriptor columns
    if non_descriptor_cols and isinstance(non_descriptor_cols, list):
        descriptor_cols = [col for col in df_analysis.columns if col not in non_descriptor_cols]
    else:
        # Fallback if non_descriptor_cols is None or empty
        descriptor_cols = [col for col in df_analysis.columns if col not in ['SMILES', 'SMILES_Valid']]
    
    if len(descriptor_cols) < 2:
        return html.Div([
            html.H5("Not enough descriptor columns", className="text-danger"),
            html.P("Please select at least 2 descriptor columns for dimensionality reduction.")
        ])
    
    # Create DataFrame with only descriptors
    descriptors_df = df_analysis[descriptor_cols].copy()
    
    # Handle missing values
    if missing_values_handling == 'drop':
        descriptors_df = descriptors_df.dropna(axis=1)
        if len(descriptors_df.columns) < 2:
            return html.Div([
                html.H5("Not enough columns remain after dropping those with missing values", className="text-danger"),
                html.P("Try another method for handling missing values.")
            ])
    elif missing_values_handling == 'mean':
        descriptors_df = descriptors_df.fillna(descriptors_df.mean())
    elif missing_values_handling == 'median':
        descriptors_df = descriptors_df.fillna(descriptors_df.median())
    else:  # zero
        descriptors_df = descriptors_df.fillna(0)
    
    # Prepare data for dimensionality reduction
    X = descriptors_df.values
    
    # Standardize if selected
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Apply dimensionality reduction
    if reduction_method == 'PCA':
        model = PCA(n_components=n_components)
        reduced_data = model.fit_transform(X)
        
        # Calculate explained variance
        explained_variance = model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Create variance plot
        fig_var = px.bar(
            x=list(range(1, len(explained_variance) + 1)),
            y=explained_variance,
            labels={'x': 'Principal components', 'y': 'Explained variance ratio'},
            title='Explained variance by components'
        )
        
        # Add cumulative variance line
        fig_var.add_scatter(
            x=list(range(1, len(cumulative_variance) + 1)),
            y=cumulative_variance,
            mode='lines+markers',
            name='Cumulative explained variance'
        )
        
        # Component loadings for PCA
        if n_components <= 10:
            loadings = model.components_
            loading_df = pd.DataFrame(loadings, columns=descriptors_df.columns)
            
            # Get the top contributing features
            pc1_loadings = loading_df.iloc[0].abs().sort_values(ascending=False).head(10)
            pc2_loadings = loading_df.iloc[1].abs().sort_values(ascending=False).head(10)
            
            # Create loadings plot
            fig_loadings = px.bar(
                x=np.concatenate([pc1_loadings.values, pc2_loadings.values]),
                y=np.concatenate([
                    [f"{col} (PC1)" for col in pc1_loadings.index],
                    [f"{col} (PC2)" for col in pc2_loadings.index]
                ]),
                orientation='h',
                title='Top Feature Contributions to PC1 and PC2',
                labels={'x': 'Absolute Loading Value', 'y': ''}
            )
        
    elif reduction_method == 't-SNE':
        model = TSNE(n_components=n_components, perplexity=tsne_perplexity, random_state=42, n_jobs=1)
        reduced_data = model.fit_transform(X)
        
    elif reduction_method == 'UMAP':
        import umap
        model = umap.UMAP(
            n_components=n_components, 
            n_neighbors=umap_n_neighbors, 
            min_dist=umap_min_dist, 
            random_state=42
        )
        reduced_data = model.fit_transform(X)
        
    elif reduction_method == 'PaCMAP' and has_pacmap:
        try:
            model = pacmap.PaCMAP(
                n_components=n_components,
                n_neighbors=pacmap_n_neighbors,
                MN_ratio=pacmap_mn_ratio,
                FP_ratio=pacmap_fp_ratio,
                random_state=42
            )
            reduced_data = model.fit_transform(X)
        except Exception as e:
            return html.Div([
                html.H5("Error during PaCMAP computation", className="text-danger"),
                html.P(str(e)),
                html.P("If this is your first time using PaCMAP, make sure it's installed with 'pip install pacmap'"),
                html.P("PaCMAP can be sensitive to dataset characteristics. You can try:"),
                html.Ul([
                    html.Li("Adjust n_neighbors (try 5-30)"),
                    html.Li("Adjust MN_ratio and FP_ratio"),
                    html.Li("Try PCA instead, which is more robust")
                ])
            ])
    else:  # Default to PCA if something goes wrong
        model = PCA(n_components=n_components)
        reduced_data = model.fit_transform(X)
        explained_variance = model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
    
    # Create a DataFrame for the reduced data
    reduced_df = pd.DataFrame(
        reduced_data, 
        columns=[f'{reduction_method} Component {i+1}' for i in range(n_components)]
    )
    
    # Add original columns needed for visualization to reduced_df
    if color_by != 'None' and color_by in df_analysis.columns:
        reduced_df[color_by] = df_analysis[color_by].values
    
    # Add SMILES column if available
    if 'SMILES' in df_analysis.columns:
        reduced_df['SMILES'] = df_analysis['SMILES'].values
    
    # Create visualization based on number of components
    if n_components == 2:
        if color_by != 'None' and color_by in reduced_df.columns:
            fig = px.scatter(
                reduced_df, 
                x=f'{reduction_method} Component 1', 
                y=f'{reduction_method} Component 2', 
                color=color_by,
                hover_data=['SMILES'] if 'SMILES' in reduced_df.columns else None,
                title=f'{reduction_method} Visualization'
            )
        else:
            fig = px.scatter(
                reduced_df, 
                x=f'{reduction_method} Component 1', 
                y=f'{reduction_method} Component 2',
                hover_data=['SMILES'] if 'SMILES' in reduced_df.columns else None,
                title=f'{reduction_method} Visualization'
            )
        
        # Create results display
        pca_specific = html.Div([
            html.H5(f"Total explained variance: {sum(explained_variance):.2%}"),
            dcc.Graph(figure=fig_var),
            html.H5("Component Loadings"),
            dbc.Table.from_dataframe(loading_df, striped=True, bordered=True, hover=True),
            dcc.Graph(figure=fig_loadings)
        ]) if reduction_method == 'PCA' else html.Div()
        
        return html.Div([
            html.H5(f"Results: {reduction_method} with {n_components} components"),
            dcc.Graph(figure=fig, id="dim-reduction-plot"),
            pca_specific,
            html.H5("Reduced Data Preview"),
            dbc.Table.from_dataframe(reduced_df.head(10), striped=True, bordered=True, hover=True),
            dbc.Button("Download Results", id="download-results-button", color="success", className="mt-3")
        ])
        
    elif n_components == 3:
        if color_by != 'None' and color_by in reduced_df.columns:
            fig = px.scatter_3d(
                reduced_df, 
                x=f'{reduction_method} Component 1', 
                y=f'{reduction_method} Component 2', 
                z=f'{reduction_method} Component 3',
                color=color_by,
                hover_data=['SMILES'] if 'SMILES' in reduced_df.columns else None,
                title=f'{reduction_method} Visualization'
            )
        else:
            fig = px.scatter_3d(
                reduced_df, 
                x=f'{reduction_method} Component 1', 
                y=f'{reduction_method} Component 2', 
                z=f'{reduction_method} Component 3',
                hover_data=['SMILES'] if 'SMILES' in reduced_df.columns else None,
                title=f'{reduction_method} Visualization'
            )
        
        # Create results display
        pca_specific = html.Div([
            html.H5(f"Total explained variance: {sum(explained_variance):.2%}"),
            dcc.Graph(figure=fig_var),
            html.H5("Component Loadings"),
            dbc.Table.from_dataframe(loading_df, striped=True, bordered=True, hover=True)
        ]) if reduction_method == 'PCA' else html.Div()
        
        return html.Div([
            html.H5(f"Results: {reduction_method} with {n_components} components"),
            dcc.Graph(figure=fig, id="dim-reduction-plot-3d"),
            pca_specific,
            html.H5("Reduced Data Preview"),
            dbc.Table.from_dataframe(reduced_df.head(10), striped=True, bordered=True, hover=True),
            dbc.Button("Download Results", id="download-results-button", color="success", className="mt-3")
        ])
        
    else:
        # Create a correlation matrix between components
        corr_matrix = reduced_df.iloc[:, :n_components].corr()
        
        # Plot correlation heatmap
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=True, 
            color_continuous_scale='Viridis',
            title='Correlation Between Components'
        )
        
        return html.Div([
            html.H5(f"Results: {reduction_method} with {n_components} components"),
            html.H5("Reduced Data Preview"),
            dbc.Table.from_dataframe(reduced_df.head(10), striped=True, bordered=True, hover=True),
            html.H5("Correlation Between Components"),
            dcc.Graph(figure=fig_corr),
            dbc.Button("Download Results", id="download-results-button", color="success", className="mt-3")
        ])

# Add download functionality
app.layout.children.append(dcc.Download(id="download-data"))

# Add storage for app data
app.layout.children.append(dcc.Store(id="app-storage"))

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

# import dash
# from dash import dcc, html, Input, Output, State, callback, ALL, MATCH
# import dash_bootstrap_components as dbc
# from dash.exceptions import PreventUpdate
# import plotly.express as px
# import pandas as pd
# import numpy as np
# import base64
# import io
# import json
# from rdkit import Chem
# from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors, AllChem, Fragments
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# import umap

# # Import PaCMAP with error handling
# try:
#     import pacmap
#     has_pacmap = True
# except ImportError:
#     has_pacmap = False

# # Initialize the Dash app with Bootstrap theme
# app = dash.Dash(__name__, 
#                 external_stylesheets=[dbc.themes.BOOTSTRAP],
#                 suppress_callback_exceptions=True)
# server = app.server  # Needed for deployment

# # Define the navbar
# navbar = dbc.NavbarSimple(
#     children=[
#         dbc.NavItem(dbc.NavLink("Data Preprocessing", href="/preprocessing")),
#         dbc.NavItem(dbc.NavLink("Analytics", href="/analytics")),
#     ],
#     brand="SMILES Data Processor",
#     brand_href="/",
#     color="primary",
#     dark=True,
# )

# # Define the layout for each page

# # Home page
# home_layout = dbc.Container([
#     dbc.Row([
#         dbc.Col(html.H1("Welcome to SMILES Data Processor"), width=12, className="mb-4 mt-4")
#     ]),
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Data Preprocessing"),
#                 dbc.CardBody([
#                     html.P("Upload data and prepare it for analysis."),
#                     dbc.Button("Go to Data Preprocessing", href="/preprocessing", color="primary")
#                 ])
#             ], className="mb-4")
#         ], width=6),
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Analytics"),
#                 dbc.CardBody([
#                     html.P("Analyze and visualize prepared data."),
#                     dbc.Button("Go to Analytics", href="/analytics", color="primary")
#                 ])
#             ], className="mb-4")
#         ], width=6)
#     ])
# ])

# # Data Preprocessing layout
# preprocessing_layout = dbc.Container([
#     dbc.Row([
#         dbc.Col(html.H1("Data Preprocessing"), width=12, className="mb-4 mt-4")
#     ]),
#     dbc.Tabs([
#         dbc.Tab([
#             dbc.Card([
#                 dbc.CardHeader("Upload your file"),
#                 dbc.CardBody([
#                     dcc.Upload(
#                         id='upload-data',
#                         children=html.Div([
#                             'Drag and Drop or ',
#                             html.A('Select Files')
#                         ]),
#                         style={
#                             'width': '100%',
#                             'height': '60px',
#                             'lineHeight': '60px',
#                             'borderWidth': '1px',
#                             'borderStyle': 'dashed',
#                             'borderRadius': '5px',
#                             'textAlign': 'center',
#                             'margin': '10px'
#                         },
#                         multiple=False
#                     ),
#                     html.Div(id='upload-output'),
#                     html.Div(id='dataframe-preview'),
#                     html.Div(id='dataset-info')
#                 ])
#             ]),
#             dbc.Card([
#                 dbc.CardHeader("Instructions"),
#                 dbc.CardBody([
#                     html.P([
#                         "1. Upload a file containing SMILES data",
#                         html.Br(),
#                         "2. The file can be one of the following formats:",
#                         html.Ul([
#                             html.Li("Excel file (.xlsx, .xls) with a column for SMILES"),
#                             html.Li("CSV file with a column for SMILES"),
#                             html.Li("Text file with one SMILES string per line")
#                         ]),
#                         "3. The app will display the dataset",
#                         html.Br(),
#                         "4. After uploading, proceed to Column Selection"
#                     ])
#                 ])
#             ], className="mt-4")
#         ], label="Data Upload"),
#         dbc.Tab([
#             dbc.Card([
#                 dbc.CardHeader("Select columns and calculate descriptors"),
#                 dbc.CardBody([
#                     html.Div(id='column-selection-container', children=[
#                         html.P("Please upload your data first in the Data Upload tab.")
#                     ]),
#                     html.Div(id='smiles-column-selection'),
#                     html.Div(id='processing-button-container'),
#                     html.Div(id='processing-output')
#                 ])
#             ])
#         ], label="Column Selection", id="column-selection-tab")
#     ])
# ])

# # Analytics layout
# analytics_layout = dbc.Container([
#     dbc.Row([
#         dbc.Col(html.H1("Analytics"), width=12, className="mb-4 mt-4")
#     ]),
#     dbc.Tabs([
#         dbc.Tab([
#             dbc.Card([
#                 dbc.CardHeader("Data Viewer"),
#                 dbc.CardBody([
#                     dbc.Tabs([
#                         dbc.Tab([
#                             dcc.Upload(
#                                 id='analytics-upload-data',
#                                 children=html.Div([
#                                     'Drag and Drop or ',
#                                     html.A('Select Files')
#                                 ]),
#                                 style={
#                                     'width': '100%',
#                                     'height': '60px',
#                                     'lineHeight': '60px',
#                                     'borderWidth': '1px',
#                                     'borderStyle': 'dashed',
#                                     'borderRadius': '5px',
#                                     'textAlign': 'center',
#                                     'margin': '10px'
#                                 },
#                                 multiple=False
#                             ),
#                             html.Div(id='analytics-upload-output'),
#                             html.Div(id='analytics-dataframe-preview'),
#                             html.Div(id='analytics-dataset-info')
#                         ], label="Upload New Data"),
#                         dbc.Tab([
#                             html.Div(id='preprocessed-data-container', children=[
#                                 html.P("Check if preprocessed data is available."),
#                                 dbc.Button("Check for Preprocessed Data", id="check-preprocessed-data", color="primary"),
#                                 html.Div(id="preprocessed-data-info")
#                             ])
#                         ], label="Use Preprocessed Data")
#                     ]),
#                     html.Div(id='data-exploration-container')
#                 ])
#             ])
#         ], label="Data Viewer"),
#         dbc.Tab([
#             dbc.Card([
#                 dbc.CardHeader("Dimensionality Reduction"),
#                 dbc.CardBody([
#                     html.Div(id='dim-reduction-container')
#                 ])
#             ])
#         ], label="Dimensionality Reduction"),
#         dbc.Tab([
#             dbc.Card([
#                 dbc.CardHeader("Clustering"),
#                 dbc.CardBody([
#                     html.P("Clustering functionality will be implemented here.")
#                 ])
#             ])
#         ], label="Clustering"),
#         dbc.Tab([
#             dbc.Card([
#                 dbc.CardHeader("Visualization"),
#                 dbc.CardBody([
#                     html.P("Visualization functionality will be implemented here.")
#                 ])
#             ])
#         ], label="Visualization")
#     ])
# ])

# # Layout for the entire app
# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     navbar,
#     html.Div(id='page-content')
# ])

# # Callback to handle page navigation
# @app.callback(
#     Output('page-content', 'children'),
#     Input('url', 'pathname')
# )
# def display_page(pathname):
#     if pathname == '/preprocessing':
#         return preprocessing_layout
#     elif pathname == '/analytics':
#         return analytics_layout
#     else:
#         return home_layout

# # Callback for file upload in Data Preprocessing
# @app.callback(
#     [Output('upload-output', 'children'),
#      Output('dataframe-preview', 'children'),
#      Output('dataset-info', 'children')],
#     Input('upload-data', 'contents'),
#     State('upload-data', 'filename')
# )
# def update_output(contents, filename):
#     if contents is None:
#         raise PreventUpdate
    
#     try:
#         # Parse uploaded file
#         content_type, content_string = contents.split(',')
#         decoded = base64.b64decode(content_string)
        
#         # Determine file type
#         if filename.endswith('.csv'):
#             # CSV file
#             df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#             success_msg = html.Div([
#                 html.H5("File successfully loaded as CSV", className="text-success")
#             ])
#         elif filename.endswith(('.xls', '.xlsx')):
#             # Excel file
#             df = pd.read_excel(io.BytesIO(decoded))
#             success_msg = html.Div([
#                 html.H5(f"File successfully loaded as Excel ({filename.split('.')[-1]})", className="text-success")
#             ])
#         else:
#             # Assume text file with one SMILES per line
#             content = decoded.decode('utf-8')
#             lines = content.strip().split('\n')
#             df = pd.DataFrame(lines, columns=['SMILES'])
#             success_msg = html.Div([
#                 html.H5("File successfully loaded as text file", className="text-success")
#             ])
        
#         # Store the dataframe in a hidden div as JSON
#         app.dataframe = df
        
#         # Create dataframe preview
#         preview = html.Div([
#             html.H5("Dataset"),
#             dbc.Table.from_dataframe(df.head(10), striped=True, bordered=True, hover=True)
#         ])
        
#         # Create dataset info
#         info = html.Div([
#             html.H5("Dataset Information"),
#             html.P(f"Number of rows: {len(df)}"),
#             html.P(f"Number of columns: {len(df.columns)}"),
#             html.P(f"Column names: {', '.join(df.columns)}")
#         ])
        
#         return success_msg, preview, info
    
#     except Exception as e:
#         return html.Div([
#             html.H5("Error processing file", className="text-danger"),
#             html.P(str(e))
#         ]), html.Div(), html.Div()

# # Callback to enable Column Selection tab after data upload
# @app.callback(
#     Output('column-selection-container', 'children'),
#     Input('upload-output', 'children')
# )
# def update_column_selection(upload_output):
#     if upload_output is None:
#         return html.P("Please upload your data first in the Data Upload tab.")
    
#     try:
#         df = app.dataframe
        
#         return html.Div([
#             html.H5("Select columns to keep"),
#             dcc.Checklist(
#                 id='column-selection-checklist',
#                 options=[{'label': col, 'value': col} for col in df.columns],
#                 value=list(df.columns),
#                 inline=False
#             )
#         ])
#     except:
#         return html.P("Please upload your data first in the Data Upload tab.")

# # Callback to add SMILES column selection
# @app.callback(
#     Output('smiles-column-selection', 'children'),
#     Input('column-selection-checklist', 'value')
# )
# def update_smiles_selection(selected_columns):
#     if not selected_columns:
#         return html.Div()
    
#     return html.Div([
#         html.H5("Select the column containing SMILES strings:", className="mt-3"),
#         dcc.Dropdown(
#             id='smiles-column-dropdown',
#             options=[{'label': col, 'value': col} for col in selected_columns],
#             value=selected_columns[0]
#         )
#     ])

# # Callback to add processing button
# @app.callback(
#     Output('processing-button-container', 'children'),
#     Input('smiles-column-dropdown', 'value')
# )
# def update_processing_button(smiles_column):
#     if not smiles_column:
#         return html.Div()
    
#     return html.Div([
#         dbc.Button(
#             "Process SMILES and Calculate Descriptors",
#             id="process-smiles-button",
#             color="primary",
#             className="mt-3"
#         )
#     ])

# # Function to calculate descriptors (simplified version of the original code)
# def calculate_descriptors(smiles):
#     """Calculate molecular descriptors for a SMILES string"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
    
#     # Basic descriptors
#     descriptors = {
#         'MW': Descriptors.ExactMolWt(mol),
#         'LogP': Descriptors.MolLogP(mol),
#         'TPSA': Descriptors.TPSA(mol),
#         'HBA': rdMolDescriptors.CalcNumHBA(mol),
#         'HBD': rdMolDescriptors.CalcNumHBD(mol),
#         'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
#         'NumAromRings': rdMolDescriptors.CalcNumAromaticRings(mol),
#         'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
#         'NumRings': rdMolDescriptors.CalcNumRings(mol)
#     }
    
#     return descriptors

# # Callback to process SMILES and calculate descriptors
# @app.callback(
#     Output('processing-output', 'children'),
#     Input('process-smiles-button', 'n_clicks'),
#     State('column-selection-checklist', 'value'),
#     State('smiles-column-dropdown', 'value')
# )
# def process_smiles(n_clicks, selected_columns, smiles_column):
#     if n_clicks is None or not selected_columns or not smiles_column:
#         return html.Div()
    
#     try:
#         df = app.dataframe.copy()
#         filtered_df = df[selected_columns].copy()
        
#         # Create progress indicators
#         progress_indicators = html.Div([
#             html.H5("Validating SMILES and Calculating Descriptors"),
#             dbc.Progress(id="smiles-progress", value=0, striped=True, animated=True)
#         ])
        
#         # Process SMILES strings and calculate descriptors
#         filtered_df['SMILES_Valid'] = False
        
#         # Lists to store valid and invalid SMILES
#         invalid_smiles = []
#         valid_smiles = []
        
#         # Dictionary to store descriptor results
#         descriptor_results = {}
        
#         # Process in chunks to keep the app responsive
#         chunk_size = 100
#         total_rows = len(filtered_df)
        
#         for i in range(0, total_rows, chunk_size):
#             chunk = filtered_df.iloc[i:min(i+chunk_size, total_rows)]
            
#             for idx, row in chunk.iterrows():
#                 smiles = row[smiles_column]
                
#                 # Skip empty strings
#                 if pd.isna(smiles) or smiles == '':
#                     invalid_smiles.append((idx, smiles, "Empty SMILES"))
#                     continue
                
#                 # Calculate descriptors
#                 descriptors = calculate_descriptors(smiles)
                
#                 if descriptors is None:
#                     # Invalid SMILES
#                     invalid_smiles.append((idx, smiles, "Invalid SMILES structure"))
#                 else:
#                     # Valid SMILES
#                     filtered_df.at[idx, 'SMILES_Valid'] = True
#                     valid_smiles.append((idx, smiles))
                    
#                     # Store descriptors
#                     for desc_name, desc_value in descriptors.items():
#                         if desc_name not in descriptor_results:
#                             descriptor_results[desc_name] = {}
#                         descriptor_results[desc_name][idx] = desc_value
        
#         # Create a new dataframe with descriptors
#         for desc_name, desc_values in descriptor_results.items():
#             filtered_df[desc_name] = pd.Series(desc_values)
        
#         # Store processed data
#         app.processed_data = filtered_df
        
#         # Create results display
#         results = html.Div([
#             html.H5("SMILES Validation Results"),
#             html.P(f"Valid SMILES: {len(valid_smiles)} of {len(filtered_df)}"),
#             html.P(f"Invalid SMILES: {len(invalid_smiles)} of {len(filtered_df)}"),
            
#             dbc.Tabs([
#                 dbc.Tab([
#                     html.Div([f"Row {idx+1}: {smiles}"] for idx, smiles in valid_smiles[:20]) if valid_smiles else html.P("No valid SMILES found.")
#                 ], label="Valid SMILES"),
#                 dbc.Tab([
#                     html.Div([f"Row {idx+1}: {smiles} - Reason: {reason}"] for idx, smiles, reason in invalid_smiles[:20]) if invalid_smiles else html.P("No invalid SMILES found.")
#                 ], label="Invalid SMILES")
#             ]),
            
#             html.H5("Processed Dataset with Descriptors", className="mt-3"),
#             dbc.Table.from_dataframe(filtered_df.head(10), striped=True, bordered=True, hover=True),
            
#             html.H5("Descriptor Statistics", className="mt-3"),
#             dbc.Table.from_dataframe(
#                 filtered_df[[col for col in filtered_df.columns if col not in selected_columns and col != 'SMILES_Valid']].describe().round(2),
#                 striped=True, bordered=True, hover=True
#             ),
            
#             dbc.Button("Download Processed Data", id="download-button", color="success", className="mt-3 me-2"),
#             dbc.Button("Download Valid Molecules Only", id="download-valid-button", color="info", className="mt-3")
#         ])
        
#         return html.Div([progress_indicators, results])
    
#     except Exception as e:
#         return html.Div([
#             html.H5("Error processing SMILES", className="text-danger"),
#             html.P(str(e))
#         ])

# # Define callback for dimensionality reduction
# @app.callback(
#     Output('dim-reduction-container', 'children'),
#     Input('url', 'pathname')
# )
# def initialize_dim_reduction(pathname):
#     if pathname != '/analytics':
#         raise PreventUpdate
    
#     # Check if data is available
#     try:
#         if hasattr(app, 'dataframe') or hasattr(app, 'processed_data'):
#             df = getattr(app, 'processed_data', None) or app.dataframe
            
#             # Create the layout for dimensionality reduction
#             return html.Div([
#                 html.H5("Select options for dimensionality reduction"),
                
#                 # Filter valid molecules if column exists
#                 dbc.Checkbox(
#                     id="use-only-valid",
#                     label="Use only valid molecules",
#                     value=True
#                 ) if 'SMILES_Valid' in df.columns else html.Div(),
                
#                 html.H5("Select columns to exclude from analysis (non-descriptors):", className="mt-3"),
#                 dcc.Dropdown(
#                     id="non-descriptor-cols",
#                     options=[{'label': col, 'value': col} for col in df.columns],
#                     value=[col for col in df.columns if col in ['SMILES', 'SMILES_Valid']],
#                     multi=True
#                 ),
                
#                 html.H5("How to handle missing values:", className="mt-3"),
#                 dcc.Dropdown(
#                     id="missing-values-handling",
#                     options=[
#                         {'label': 'Drop columns with missing values', 'value': 'drop'},
#                         {'label': 'Fill missing values with mean', 'value': 'mean'},
#                         {'label': 'Fill missing values with median', 'value': 'median'},
#                         {'label': 'Fill missing values with 0', 'value': 'zero'}
#                     ],
#                     value='drop'
#                 ),
                
#                 dbc.Checkbox(
#                     id="standardize-data",
#                     label="Standardize data (recommended)",
#                     value=True,
#                     className="mt-3"
#                 ),
                
#                 html.H5("Select dimensionality reduction method:", className="mt-3"),
#                 dcc.Dropdown(
#                     id="reduction-method",
#                     options=[
#                         {'label': 'PCA', 'value': 'PCA'},
#                         {'label': 't-SNE', 'value': 't-SNE'},
#                         {'label': 'UMAP', 'value': 'UMAP'},
#                         {'label': 'PaCMAP', 'value': 'PaCMAP'} if has_pacmap else None
#                     ],
#                     value='PCA'
#                 ),
#                 html.Div(id="pacmap-status", children=[
#                     html.P("PaCMAP is not available. Install with 'pip install pacmap'", 
#                            className="text-warning") if not has_pacmap else None
#                 ]),
                
#                 html.Div(id="method-specific-params"),
                
#                 html.H5("Number of components:", className="mt-3"),
#                 dcc.Slider(
#                     id="n-components",
#                     min=2,
#                     max=min(10, len([col for col in df.columns if col not in ['SMILES', 'SMILES_Valid']])),
#                     step=1,
#                     value=2,
#                     marks={i: str(i) for i in range(2, 11)}
#                 ),
                
#                 html.H5("Color points by:", className="mt-3"),
#                 dcc.Dropdown(
#                     id="color-by",
#                     options=[{'label': 'None', 'value': 'None'}] + 
#                             [{'label': col, 'value': col} for col in df.columns],
#                     value='None'
#                 ),
                
#                 dbc.Button(
#                     "Run Dimensionality Reduction",
#                     id="run-dim-reduction",
#                     color="primary",
#                     className="mt-3"
#                 ),
                
#                 html.Div(id="dim-reduction-results")
#             ])
#         else:
#             return html.Div([
#                 html.H5("No data available for analysis."),
#                 html.P("Please upload data in the Data Viewer or process data in the Data Preprocessing section first.")
#             ])
#     except:
#         return html.Div([
#             html.H5("No data available for analysis."),
#             html.P("Please upload data in the Data Viewer or process data in the Data Preprocessing section first.")
#         ])

# # Add method-specific parameters
# @app.callback(
#     Output('method-specific-params', 'children'),
#     Input('reduction-method', 'value')
# )
# def update_method_params(method):
#     if method == 't-SNE':
#         return html.Div([
#             html.H5("Perplexity:", className="mt-3"),
#             dcc.Slider(
#                 id="tsne-perplexity",
#                 min=5,
#                 max=50,
#                 step=1,
#                 value=30,
#                 marks={i: str(i) for i in range(5, 51, 5)}
#             )
#         ])
#     elif method == 'UMAP':
#         return html.Div([
#             html.H5("Number of neighbors:", className="mt-3"),
#             dcc.Slider(
#                 id="umap-n-neighbors",
#                 min=2,
#                 max=100,
#                 step=1,
#                 value=15,
#                 marks={i: str(i) for i in [2, 5, 10, 15, 20, 30, 50, 100]}
#             ),
            
#             html.H5("Minimum distance:", className="mt-3"),
#             dcc.Slider(
#                 id="umap-min-dist",
#                 min=0,
#                 max=0.99,
#                 step=0.01,
#                 value=0.1,
#                 marks={i/100: str(i/100) for i in range(0, 100, 10)}
#             )
#         ])
#     elif method == 'PaCMAP' and has_pacmap:
#         return html.Div([
#             html.H5("Number of neighbors:", className="mt-3"),
#             dcc.Slider(
#                 id="pacmap-n-neighbors",
#                 min=5,
#                 max=100,
#                 step=1,
#                 value=10,
#                 marks={i: str(i) for i in [5, 10, 20, 30, 50, 100]}
#             ),
            
#             html.H5("MN ratio:", className="mt-3"),
#             dcc.Slider(
#                 id="pacmap-mn-ratio",
#                 min=0.1,
#                 max=1.0,
#                 step=0.1,
#                 value=0.5,
#                 marks={i/10: str(i/10) for i in range(1, 11)},
#                 tooltip={"placement": "bottom", "always_visible": True}
#             ),
            
#             html.H5("FP ratio:", className="mt-3"),
#             dcc.Slider(
#                 id="pacmap-fp-ratio",
#                 min=0.1,
#                 max=5.0,
#                 step=0.1,
#                 value=3.0,
#                 marks={i: str(i) for i in range(0, 6)},
#                 tooltip={"placement": "bottom", "always_visible": True}
#             ),
            
#             html.P("PaCMAP works best with standardized data and may take some time for larger datasets.",
#                   className="text-info mt-3")
#         ])
#     else:  # PCA
#         return html.Div()

# # Run dimensionality reduction
# @app.callback(
#     Output('dim-reduction-results', 'children'),
#     Input('run-dim-reduction', 'n_clicks'),
#     [State('use-only-valid', 'value'),
#      State('non-descriptor-cols', 'value'),
#      State('missing-values-handling', 'value'),
#      State('standardize-data', 'value'),
#      State('reduction-method', 'value'),
#      State('n-components', 'value'),
#      State('color-by', 'value'),
#      State('tsne-perplexity', 'value'),
#      State('umap-n-neighbors', 'value'),
#      State('umap-min-dist', 'value'),
#      State('pacmap-n-neighbors', 'value'),
#      State('pacmap-mn-ratio', 'value'),
#      State('pacmap-fp-ratio', 'value')]
# )
# def run_dimensionality_reduction(n_clicks, use_only_valid, non_descriptor_cols, missing_values_handling, 
#                                standardize, reduction_method, n_components, color_by, 
#                                tsne_perplexity, umap_n_neighbors, umap_min_dist,
#                                pacmap_n_neighbors, pacmap_mn_ratio, pacmap_fp_ratio):
#     if n_clicks is None:
#         raise PreventUpdate
    
#     try:
#         # Get the data
#         df = getattr(app, 'processed_data', None) or app.dataframe
        
#         # Filter to only valid molecules if desired and if the column exists
#         if use_only_valid and 'SMILES_Valid' in df.columns:
#             df_analysis = df[df['SMILES_Valid'] == True].copy()
#             if df_analysis.empty:
#                 return html.Div([
#                     html.H5("No valid molecules found.", className="text-warning"),
#                     html.P("Please adjust your filter or use all molecules.")
#                 ])
#         else:
#             df_analysis = df.copy()
        
#         # Get descriptor columns
#         descriptor_cols = [col for col in df_analysis.columns if col not in non_descriptor_cols]
        
#         if len(descriptor_cols) < 2:
#             return html.Div([
#                 html.H5("Not enough descriptor columns", className="text-danger"),
#                 html.P("Please select at least 2 descriptor columns for dimensionality reduction.")
#             ])
        
#         # Create DataFrame with only descriptors
#         descriptors_df = df_analysis[descriptor_cols].copy()
        
#         # Handle missing values
#         if missing_values_handling == 'drop':
#             descriptors_df = descriptors_df.dropna(axis=1)
#             if len(descriptors_df.columns) < 2:
#                 return html.Div([
#                     html.H5("Not enough columns remain after dropping those with missing values.", className="text-danger"),
#                     html.P("Try another method for handling missing values.")
#                 ])
#         elif missing_values_handling == 'mean':
#             descriptors_df = descriptors_df.fillna(descriptors_df.mean())
#         elif missing_values_handling == 'median':
#             descriptors_df = descriptors_df.fillna(descriptors_df.median())
#         else:  # zero
#             descriptors_df = descriptors_df.fillna(0)
        
#         # Prepare data for dimensionality reduction
#         X = descriptors_df.values
        
#         # Standardize if selected
#         if standardize:
#             scaler = StandardScaler()
#             X = scaler.fit_transform(X)
        
#         # Apply dimensionality reduction
#         if reduction_method == 'PCA':
#             model = PCA(n_components=n_components)
#             reduced_data = model.fit_transform(X)
            
#             # Calculate explained variance
#             explained_variance = model.explained_variance_ratio_
#             cumulative_variance = np.cumsum(explained_variance)
            
#             # Create variance plot
#             fig_var = px.bar(
#                 x=list(range(1, len(explained_variance) + 1)),
#                 y=explained_variance,
#                 labels={'x': 'Principal components', 'y': 'Explained variance ratio'},
#                 title='Explained variance by components'
#             )
            
#             # Add cumulative variance line
#             fig_var.add_scatter(
#                 x=list(range(1, len(cumulative_variance) + 1)),
#                 y=cumulative_variance,
#                 mode='lines+markers',
#                 name='Cumulative explained variance'
#             )
            
#             # Component loadings for PCA
#             if n_components <= 10:
#                 loadings = model.components_
#                 loading_df = pd.DataFrame(loadings, columns=descriptors_df.columns)
                
#                 # Get the top contributing features
#                 pc1_loadings = loading_df.iloc[0].abs().sort_values(ascending=False).head(10)
#                 pc2_loadings = loading_df.iloc[1].abs().sort_values(ascending=False).head(10)
                
#                 # Create loadings plot
#                 fig_loadings = px.bar(
#                     x=np.concatenate([pc1_loadings.values, pc2_loadings.values]),
#                     y=np.concatenate([
#                         [f"{col} (PC1)" for col in pc1_loadings.index],
#                         [f"{col} (PC2)" for col in pc2_loadings.index]
#                     ]),
#                     orientation='h',
#                     title='Top Feature Contributions to PC1 and PC2',
#                     labels={'x': 'Absolute Loading Value', 'y': ''}
#                 )
            
#         elif reduction_method == 't-SNE':
#             model = TSNE(n_components=n_components, perplexity=tsne_perplexity, random_state=42, n_jobs=1)
#             reduced_data = model.fit_transform(X)
            
#         elif reduction_method == 'UMAP':
#             import umap
#             model = umap.UMAP(
#                 n_components=n_components, 
#                 n_neighbors=umap_n_neighbors, 
#                 min_dist=umap_min_dist, 
#                 random_state=42
#             )
#             reduced_data = model.fit_transform(X)
            
#         elif reduction_method == 'PaCMAP' and has_pacmap:
#             try:
#                 model = pacmap.PaCMAP(
#                     n_components=n_components,
#                     n_neighbors=pacmap_n_neighbors,
#                     MN_ratio=pacmap_mn_ratio,
#                     FP_ratio=pacmap_fp_ratio,
#                     random_state=42
#                 )
#                 reduced_data = model.fit_transform(X)
#             except Exception as e:
#                 return html.Div([
#                     html.H5("Error during PaCMAP computation", className="text-danger"),
#                     html.P(str(e)),
#                     html.P("If this is your first time using PaCMAP, make sure it's installed with 'pip install pacmap'"),
#                     html.P("PaCMAP can be sensitive to dataset characteristics. You can try:"),
#                     html.Ul([
#                         html.Li("Adjust n_neighbors (try 5-30)"),
#                         html.Li("Adjust MN_ratio and FP_ratio"),
#                         html.Li("Try PCA instead, which is more robust")
#                     ])
#                 ])
#         else:  # Default to PCA if something goes wrong
#             model = PCA(n_components=n_components)
#             reduced_data = model.fit_transform(X)
#             explained_variance = model.explained_variance_ratio_
#             cumulative_variance = np.cumsum(explained_variance)
        
#         # Create a DataFrame for the reduced data
#         reduced_df = pd.DataFrame(
#             reduced_data, 
#             columns=[f'{reduction_method} Component {i+1}' for i in range(n_components)]
#         )
        
#         # Add original columns needed for visualization to reduced_df
#         if color_by != 'None' and color_by in df_analysis.columns:
#             reduced_df[color_by] = df_analysis[color_by].values
        
#         # Create visualization based on number of components
#         if n_components == 2:
#             if color_by != 'None' and color_by in reduced_df.columns:
#                 fig = px.scatter(
#                     reduced_df, 
#                     x=f'{reduction_method} Component 1', 
#                     y=f'{reduction_method} Component 2', 
#                     color=color_by,
#                     hover_data=df_analysis.columns if 'SMILES' in df_analysis.columns else None,
#                     title=f'{reduction_method} Visualization'
#                 )
#             else:
#                 fig = px.scatter(
#                     reduced_df, 
#                     x=f'{reduction_method} Component 1', 
#                     y=f'{reduction_method} Component 2',
#                     hover_data=df_analysis.columns if 'SMILES' in df_analysis.columns else None,
#                     title=f'{reduction_method} Visualization'
#                 )
            
#             # Create results display
#             pca_specific = html.Div([
#                 html.H5(f"Total explained variance: {sum(explained_variance):.2%}"),
#                 dcc.Graph(figure=fig_var),
#                 html.H5("Component Loadings"),
#                 dbc.Table.from_dataframe(loading_df, striped=True, bordered=True, hover=True),
#                 dcc.Graph(figure=fig_loadings)
#             ]) if reduction_method == 'PCA' else html.Div()
            
#             return html.Div([
#                 html.H5(f"Results: {reduction_method} with {n_components} components"),
#                 dcc.Graph(figure=fig),
#                 pca_specific,
#                 html.H5("Reduced Data Preview"),
#                 dbc.Table.from_dataframe(reduced_df.head(10), striped=True, bordered=True, hover=True),
#                 dbc.Button("Download Results", id="download-results-button", color="success", className="mt-3")
#             ])
            
#         elif n_components == 3:
#             if color_by != 'None' and color_by in reduced_df.columns:
#                 fig = px.scatter_3d(
#                     reduced_df, 
#                     x=f'{reduction_method} Component 1', 
#                     y=f'{reduction_method} Component 2', 
#                     z=f'{reduction_method} Component 3',
#                     color=color_by,
#                     hover_data=df_analysis.columns if 'SMILES' in df_analysis.columns else None,
#                     title=f'{reduction_method} Visualization'
#                 )
#             else:
#                 fig = px.scatter_3d(
#                     reduced_df, 
#                     x=f'{reduction_method} Component 1', 
#                     y=f'{reduction_method} Component 2', 
#                     z=f'{reduction_method} Component 3',
#                     hover_data=df_analysis.columns if 'SMILES' in df_analysis.columns else None,
#                     title=f'{reduction_method} Visualization'
#                 )
            
#             # Create results display
#             pca_specific = html.Div([
#                 html.H5(f"Total explained variance: {sum(explained_variance):.2%}"),
#                 dcc.Graph(figure=fig_var),
#                 html.H5("Component Loadings"),
#                 dbc.Table.from_dataframe(loading_df, striped=True, bordered=True, hover=True)
#             ]) if reduction_method == 'PCA' else html.Div()
            
#             return html.Div([
#                 html.H5(f"Results: {reduction_method} with {n_components} components"),
#                 dcc.Graph(figure=fig),
#                 pca_specific,
#                 html.H5("Reduced Data Preview"),
#                 dbc.Table.from_dataframe(reduced_df.head(10), striped=True, bordered=True, hover=True),
#                 dbc.Button("Download Results", id="download-results-button", color="success", className="mt-3")
#             ])
            
#         else:
#             # Create a correlation matrix between components
#             corr_matrix = reduced_df.iloc[:, :n_components].corr()
            
#             # Plot correlation heatmap
#             fig_corr = px.imshow(
#                 corr_matrix, 
#                 text_auto=True, 
#                 color_continuous_scale='Viridis',
#                 title='Correlation Between Components'
#             )
            
#             return html.Div([
#                 html.H5(f"Results: {reduction_method} with {n_components} components"),
#                 html.H5("Reduced Data Preview"),
#                 dbc.Table.from_dataframe(reduced_df.head(10), striped=True, bordered=True, hover=True),
#                 html.H5("Correlation Between Components"),
#                 dcc.Graph(figure=fig_corr),
#                 dbc.Button("Download Results", id="download-results-button", color="success", className="mt-3")
#             ])
    
#     except Exception as e:
#         return html.Div([
#             html.H5("Error during dimensionality reduction", className="text-danger"),
#             html.P(str(e)),
#             html.P("Try using fewer components, different parameters, or a different method.")
#         ])

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)