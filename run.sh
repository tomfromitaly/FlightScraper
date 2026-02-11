#!/bin/bash

# Flight Scraper - Streamlit App Runner
# This script activates the virtual environment and runs the Streamlit app

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Run the Streamlit app
streamlit run "$SCRIPT_DIR/app.py"
