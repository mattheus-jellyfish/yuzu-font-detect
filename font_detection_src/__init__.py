"""
Font Detection Source Package
=============================

This package contains the core font detection functionality including:
- complete_font_detection.py: Main font detection script with model definitions
- fonts.py: List of fonts the model was trained on
- Model checkpoint files (.ckpt)
"""

from .complete_font_detection import detect_fonts, load_font_detection_model
from .fonts import FONT_LIST

__all__ = ['detect_fonts', 'load_font_detection_model', 'FONT_LIST'] 