# File: project2/pages/hia.py

import streamlit as st
import sys
import os

# Add the 'project2' root directory to the Python path
# This allows Python to find 'project1_core'
# os.path.dirname(__file__) is 'project2/pages'
# os.path.dirname(os.path.dirname(__file__)) is 'project2'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    # --- IMPORTANT CHANGE HERE ---
    # Now that main.py is inside 'src', the import path must include 'src'
    from project1_core.src.main import run_hia_dashboard
except ImportError as e:
    st.error(f"Error loading HIA Dashboard content: {e}. Please ensure 'project1_core' is a package, and its internal imports are correct.")
    st.stop()

st.title("💡 Health Insights Agent (HIA)")
st.write("This page provides the interactive dashboard and analysis features.")

# Call the encapsulated function to run Project 1's Streamlit logic
run_hia_dashboard()