#!/bin/bash

# Helper script to run the traffic perception pipeline
# Activates the main project venv automatically

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate the virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Run yolo_inference.py with all passed arguments
cd "$SCRIPT_DIR"
python yolo_inference.py "$@"
