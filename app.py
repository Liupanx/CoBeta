"""
Main entry point for Hugging Face Spaces deployment.
This file is required by Hugging Face Spaces to launch the Gradio app.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.beta.gradio_app import create_interface

# Create and launch the interface
demo = create_interface()
demo.launch()

