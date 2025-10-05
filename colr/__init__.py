"""
Palette Extractor - A Python package for extracting color palettes from images.

This package provides tools to extract dominant colors from images using 
K-means clustering with optional image pooling and color diversity filtering.
"""

__version__ = "0.1.0"
__author__ = "tiam4tt"
__email__ = "khuuthanhthien269@gmail.com"

from .extractor import ColorPaletteExtractor
from .exceptions import PaletteExtractorError, ImageLoadError, InvalidConfigurationError, InsufficientColorsError

__all__ = [
    "ColorPaletteExtractor",
    "PaletteExtractorError", 
    "ImageLoadError",
    "InvalidConfigurationError",
    "InsufficientColorsError"
]