import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

# Load results folder
folder = r'RESULTS\2025-03-06_12-55-55-cmc'

original_data = pd.read_csv(f'{folder}\\original_data.csv')
reduced_data = pd.read_csv(f'{folder}\\reduced_data.csv')
cluster_labels = pd.read_csv(f'{folder}\\cluster_labels.csv')

# print(reduced_data[['ID']].head())
# print(original_data.head())

# Ensure reduced_data has an Index column for mapping
reduced_data['ID'] = reduced_data.index
cluster_labels['ID'] = cluster_labels.index

# # Merge cluster labels into reduced_data for easy reference
# if 'Cluster' not in reduced_data.columns:
#     reduced_data = reduced_data.merge(cluster_labels, on='ID', how='left')

# Create the Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive DR & Clustering"),
    
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
    
    # Table to display original data points on selection
    html.Div(id='selected-data-table'),
    
    # Display original data when hovering
    html.Div(id='hover-data')
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('color-option', 'value')
)
def update_scatter_plot(color_option):
    # # Attempt #1: Does not work. Each trace seems to reset its ID values. Only color "0" worked for the mapping.
    # if color_option == 'cluster':
    #     fig = px.scatter(
    #         reduced_data, x='Dim1', y='Dim2',
    #         # color=cluster_labels['Cluster Labels'].astype(str),  # Ensure categorical colors
    #         hover_data={'ID': True},
    #         custom_data=['ID']
    #     )
    # Attempt #2: Works. Used the column directly from reduced_data. However, the color scale is not categorical.
    if color_option == 'cluster':
        fig = px.scatter(
            reduced_data, 
            x='Dim1', y='Dim2',
            # color='Cluster Labels',  # Use the column directly from reduced_data
            color='Label',
            hover_data={'ID': True},
            custom_data=['ID'],
            color_continuous_scale=px.colors.sequential.Turbo      # Use a categorical color scale
        )
    # Attempt #3: #TODO: Fix the color scale to be categorical
    # had issues with the mapping getting messed up when doing this. see attempt #1
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
        fig.update_traces(marker=dict(size=5, color='blue'))
    
    fig.update_traces(
        selected=dict(marker=dict(color='red')),
        unselected=dict(marker=dict(color='blue'))
    )
    fig.update_layout(uirevision='constant')
    return fig

@app.callback(
    Output('selected-data-table', 'children'),
    Input('scatter-plot', 'selectedData')
)
def update_table(selectedData):
    if selectedData and 'points' in selectedData:
        selected_indices = []
        for point in selectedData['points']:
            if 'customdata' in point and point['customdata']:
                selected_indices.append(point['customdata'][0])
                selected_indices = [i - 1 for i in selected_indices]
            elif 'pointIndex' in point:
                selected_indices.append(point['pointIndex'])
                selected_indices = [i - 1 for i in selected_indices]
        if not selected_indices:
            return html.Div("No valid point indices found in selection.")
        
        selected_df = original_data.iloc[selected_indices]
        return html.Div([
            html.H3(f"Selected {len(selected_indices)} Points in Original Dataset"),
            dash_table.DataTable(
                data=selected_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in selected_df.columns],
                style_table={'overflowX': 'auto'},
                page_size=10
            )
        ])
    return html.Div("Select multiple points using the box or lasso tool to view original values.")


@app.callback(
    Output('hover-data', 'children'),
    Input('scatter-plot', 'hoverData')
)
def display_hover_data(hoverData):
    if hoverData and 'points' in hoverData and hoverData['points']:
        point_info = hoverData['points'][0]
        # Attempt to get the point index from 'customdata' if available.
        if 'customdata' in point_info and point_info['customdata']:
            point_index = point_info['customdata'][0]
            point_index = point_index - 1
        # Fallback to 'pointIndex' if 'customdata' is missing.
        elif 'pointIndex' in point_info:
            point_index = point_info['pointIndex']
            point_index = point_index - 1
        else:
            return html.Div("Hovered point data is missing expected information.")
        
        original_name = original_data.iloc[point_index]['Name']
        hover_df = original_data.iloc[[point_index]]
        return html.Div([
            html.H3(f"Hovered Point Mapping: {original_name}"),
            dash_table.DataTable(
                data=hover_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in hover_df.columns],
                style_table={'overflowX': 'auto'}
            )
        ])
    return html.Div("Hover over a point to see its original values.")


if __name__ == '__main__':
    app.run_server(debug=True)
