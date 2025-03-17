import streamlit as st
import subprocess
import time
import socket
import os
import sys
from pathlib import Path
import webbrowser

# Set page config
st.set_page_config(
    page_title="Molecule Visualization Dashboard",
    layout="wide"
)

# Function to check if port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Function to find a free port
def find_free_port(start_port=8050):
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

# Title and description
st.title("Interactive Molecule Visualization Dashboard")
st.markdown("""
This application launches a Dash server for advanced molecule visualization capabilities.
The Dash component handles the interactive selection and visualization of molecules.
""")

# Create a temporary script file to run the Dash app
def create_dash_script(folder_path, port):
    script_content = f"""
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.getcwd())

try:
    from dash_molecule_app import create_dash_app
    
    # Create and run the Dash app
    app = create_dash_app(folder_path="{folder_path}", port={port})
    
    if app:
        app.run_server(debug=False, port={port})
    else:
        print("Failed to create Dash app. Check the folder path and data files.")
except Exception as e:
    print(f"Error starting Dash app: {{e}}")
"""
    
    # Create a temporary script file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    script_path = temp_dir / "run_dash_app.py"
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    return str(script_path)

# Initialize session state for the Dash process
if 'dash_process' not in st.session_state:
    st.session_state.dash_process = None
if 'port' not in st.session_state:
    st.session_state.port = None

# Sidebar for settings
with st.sidebar:
    st.header("Dashboard Settings")
    
    # Input for results folder
    folder_path = st.text_input(
        "Results Folder Path", 
        value="RESULTS/2025-03-09_20-03-29-logcmc",
        help="Path to folder containing the data files (original_data.csv, reduced_data.csv, etc.)"
    )
    
    # Button to launch/reload the Dash app
    if st.button("Launch/Reload Dashboard"):
        # Cleanup any existing Dash process
        if st.session_state.dash_process:
            try:
                st.session_state.dash_process.terminate()
            except:
                pass
            st.session_state.dash_process = None
        
        # Find a free port
        port = find_free_port()
        st.session_state.port = port
        
        # Create a temporary script file
        script_path = create_dash_script(folder_path, port)
        
        # Start the Dash server as a subprocess
        try:
            # Hide console window on Windows
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo
            )
            
            st.session_state.dash_process = process
            
            # Wait for server to start
            time.sleep(3)
            
            # Check if server started successfully
            if is_port_in_use(port):
                st.success(f"Dashboard started on port {port}")
            else:
                st.error("Failed to start dashboard. Check the console for errors.")
                if st.session_state.dash_process:
                    stdout, stderr = st.session_state.dash_process.communicate(timeout=1)
                    st.code(stderr.decode())
        except Exception as e:
            st.error(f"Error starting Dash server: {e}")
    
    # Button to stop the Dash server
    if st.button("Stop Dashboard"):
        if st.session_state.dash_process:
            try:
                st.session_state.dash_process.terminate()
            except:
                pass
            st.session_state.dash_process = None
            st.session_state.port = None
            st.success("Dashboard stopped")
        else:
            st.warning("No dashboard is currently running")

# Main content
if st.session_state.port and is_port_in_use(st.session_state.port):
    # Display URL to open Dash app
    dash_url = f"http://localhost:{st.session_state.port}"
    
    st.success(f"✅ Dashboard is running!")
    st.markdown(f"### [Click here to open the Dashboard in a new tab]({dash_url})")
    
    # Added iframe embed using markdown (works better on more Streamlit versions)
    st.markdown(
        f"""
        <div style="height:700px; margin-top:20px;">
            <iframe src="{dash_url}" width="100%" height="100%" frameborder="0">
            </iframe>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Provide additional instructions
    st.info("If the embedded view above doesn't work well, please use the link to open in a new tab.")
else:
    st.info("Click 'Launch Dashboard' in the sidebar to start the interactive visualization.")

# Instructions
with st.expander("How to Use This Dashboard"):
    st.markdown("""
    ### Selection Tools
    1. Use the lasso or box select tools from the plotly toolbar
    2. Draw around points to select them
    3. The selected molecules and their data will automatically appear below the plot
    
    ### Color Options
    - **Default**: All points shown in blue
    - **Cluster Labels**: Points colored by their cluster assignment
    - **Target Values**: Points colored by their target property value
    
    ### Data Exploration
    - Hover over points to see details
    - Select multiple points to compare their structures and properties
    """)

# Display dataset information
with st.expander("About the Dataset"):
    st.markdown("""
    This dashboard visualizes molecular data with the following components:
    
    - **original_data.csv**: Contains the original molecular properties
    - **reduced_data.csv**: Contains the dimensionality-reduced coordinates (e.g., PCA, t-SNE)
    - **cluster_labels.csv**: Contains the cluster assignments
    - **original_data_SMILES.csv**: Contains the SMILES strings for molecular rendering
    
    The visualization allows you to explore relationships between molecular structure and properties.
    """)

# Handle cleanup when the app is closed
def cleanup():
    if st.session_state.dash_process:
        try:
            st.session_state.dash_process.terminate()
        except:
            pass

# Register the cleanup function to be called when Streamlit exits
import atexit
atexit.register(cleanup)

# import streamlit as st
# import subprocess
# import time
# import os
# import sys
# import signal
# from dash_molecule_app import create_dash_app
# import threading
# import psutil

# # Set page config
# st.set_page_config(
#     page_title="Molecule Visualization Dashboard",
#     layout="wide"
# )

# # Function to check if port is in use
# def is_port_in_use(port):
#     import socket
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         return s.connect_ex(('localhost', port)) == 0

# # Function to find a free port
# def find_free_port(start_port=8050):
#     port = start_port
#     while is_port_in_use(port):
#         port += 1
#     return port

# # Function to start Dash server in a separate process
# def start_dash_server(folder_path, port):
#     from multiprocessing import Process
#     import dash_molecule_app
    
#     def run_dash():
#         app = dash_molecule_app.create_dash_app(folder_path, port)
#         app.run_server(debug=False, port=port)
    
#     process = Process(target=run_dash)
#     process.daemon = True
#     process.start()
#     return process

# # Function to stop all Dash servers
# def stop_dash_servers():
#     # Find all Python processes
#     for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
#         try:
#             # Check if it's a Dash server
#             if proc.info['name'] == 'python' and any('dash' in arg.lower() for arg in proc.info['cmdline'] if isinstance(arg, str)):
#                 st.write(f"Terminating Dash process with PID {proc.info['pid']}")
#                 psutil.Process(proc.info['pid']).terminate()
#         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
#             pass

# # Initialize session state for the Dash process
# if 'dash_process' not in st.session_state:
#     st.session_state.dash_process = None
# if 'port' not in st.session_state:
#     st.session_state.port = None

# # Title and description
# st.title("Interactive Molecule Visualization Dashboard")
# st.markdown("""
# This application uses Dash embedded in Streamlit to provide advanced molecule visualization capabilities.
# The Dash component handles the interactive selection and visualization of molecules.
# """)

# # Sidebar for settings
# with st.sidebar:
#     st.header("Dashboard Settings")
    
#     # Input for results folder
#     folder_path = st.text_input(
#         "Results Folder Path", 
#         value="RESULTS/2025-03-09_20-03-29-logcmc",
#         help="Path to folder containing the data files (original_data.csv, reduced_data.csv, etc.)"
#     )
    
#     # Button to launch/reload the Dash app
#     if st.button("Launch/Reload Dashboard"):
#         # Cleanup any existing Dash process
#         if st.session_state.dash_process:
#             st.session_state.dash_process.terminate()
#             st.session_state.dash_process = None
        
#         # Find a free port
#         port = find_free_port()
#         st.session_state.port = port
        
#         # Start the Dash server
#         st.session_state.dash_process = start_dash_server(folder_path, port)
        
#         # Wait for server to start
#         time.sleep(2)
#         st.success(f"Dashboard started on port {port}")
    
#     # Button to stop the Dash server
#     if st.button("Stop Dashboard"):
#         if st.session_state.dash_process:
#             st.session_state.dash_process.terminate()
#             st.session_state.dash_process = None
#             st.session_state.port = None
#             st.success("Dashboard stopped")
#         else:
#             st.warning("No dashboard is currently running")

# # Main content
# if st.session_state.port:
#     # Embed the Dash app in an iframe
#     st.components.html(
#         f"""
#         <div style="height: 1000px; width: 100%; overflow: hidden;">
#             <iframe src="http://localhost:{st.session_state.port}" 
#                     style="position: relative; width: 100%; height: 100%; border: none;">
#             </iframe>
#         </div>
#         """,
#         height=1000,
#     )
# else:
#     st.info("Click 'Launch Dashboard' in the sidebar to start the interactive visualization.")

# # Instructions
# with st.expander("How to Use This Dashboard"):
#     st.markdown("""
#     ### Selection Tools
#     1. Use the lasso or box select tools from the plotly toolbar
#     2. Draw around points to select them
#     3. The selected molecules and their data will automatically appear below the plot
    
#     ### Color Options
#     - **Default**: All points shown in blue
#     - **Cluster Labels**: Points colored by their cluster assignment
#     - **Target Values**: Points colored by their target property value
    
#     ### Data Exploration
#     - Hover over points to see details
#     - Select multiple points to compare their structures and properties
#     """)

# # Display dataset information
# with st.expander("About the Dataset"):
#     st.markdown("""
#     This dashboard visualizes molecular data with the following components:
    
#     - **original_data.csv**: Contains the original molecular properties
#     - **reduced_data.csv**: Contains the dimensionality-reduced coordinates (e.g., PCA, t-SNE)
#     - **cluster_labels.csv**: Contains the cluster assignments
#     - **original_data_SMILES.csv**: Contains the SMILES strings for molecular rendering
    
#     The visualization allows you to explore relationships between molecular structure and properties.
#     """)

# # Handle cleanup when the app is closed
# def cleanup():
#     if st.session_state.dash_process:
#         st.session_state.dash_process.terminate()

# # Register the cleanup function to be called when Streamlit exits
# import atexit
# atexit.register(cleanup)