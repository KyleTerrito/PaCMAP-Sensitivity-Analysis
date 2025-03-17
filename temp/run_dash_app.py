
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.getcwd())

try:
    from dash_molecule_app import create_dash_app
    
    # Create and run the Dash app
    app = create_dash_app(folder_path="RESULTS/2025-03-09_20-03-29-logcmc", port=8050)
    
    if app:
        app.run_server(debug=False, port=8050)
    else:
        print("Failed to create Dash app. Check the folder path and data files.")
except Exception as e:
    print(f"Error starting Dash app: {e}")
