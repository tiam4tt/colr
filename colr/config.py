"""
Configuration constants and default values for the palette extractor.
"""

from typing import List, Literal

# Default extraction parameters
DEFAULT_WINDOW_SIZE: int = 4
DEFAULT_STRIDE: int = 4
DEFAULT_POOLING_METHOD: Literal["mean", "max", "min"] = "mean"
DEFAULT_K_COLORS: int = 6
DEFAULT_DIVERSITY_FACTOR: float = 0.4
DEFAULT_K_MEANS_INIT: int = 5
DEFAULT_RANDOM_STATE: int = 42

# Color calculation constants
BRIGHTNESS_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)

# File output defaults
DEFAULT_JSON_FILENAME: str = "palette.json"
DEFAULT_PNG_FILENAME: str = "palette.png"

# Image output settings
PALETTE_COLOR_WIDTH: int = 120
PALETTE_IMAGE_HEIGHT: int = 140
PALETTE_COLOR_RECT_HEIGHT: int = 80
PALETTE_MARGIN: int = 5
PALETTE_TEXT_Y_POSITIONS: tuple[int, int, int] = (95, 110, 125)

# Supported pooling methods
POOLING_METHODS: List[Literal["mean", "max", "min"]] = ["mean", "max", "min"]

# Minimum distance threshold for color diversity
MIN_DIVERSITY_DISTANCE: float = 100.0

# Text color brightness threshold
TEXT_COLOR_BRIGHTNESS_THRESHOLD: int = 128