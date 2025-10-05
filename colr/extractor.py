"""
Core color palette extraction functionality.

This module contains the main ColorPaletteExtractor class that handles
image processing, color clustering, and palette generation.
"""

import json
import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from .config import (
    BRIGHTNESS_WEIGHTS,
    DEFAULT_DIVERSITY_FACTOR,
    DEFAULT_K_COLORS,
    DEFAULT_K_MEANS_INIT,
    DEFAULT_POOLING_METHOD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    MIN_DIVERSITY_DISTANCE,
    PALETTE_COLOR_RECT_HEIGHT,
    PALETTE_COLOR_WIDTH,
    PALETTE_IMAGE_HEIGHT,
    PALETTE_MARGIN,
    PALETTE_TEXT_Y_POSITIONS,
    TEXT_COLOR_BRIGHTNESS_THRESHOLD,
)
from .exceptions import ImageLoadError, InvalidConfigurationError


class ColorPaletteExtractor:
    """
    Extract color palettes from images using K-means clustering.
    
    This class provides functionality to load images, apply optional pooling
    for dimensionality reduction, and extract dominant colors using K-means
    clustering with diversity filtering.
    
    Args:
        image_path: Path to the input image file
        window_size: Size of the pooling window (default: 4)
        stride: Stride for pooling operation (default: 4)
        pooling_method: Method for pooling ('mean', 'max', or 'min')
        darkmode: Whether to use dark mode (swap background/foreground)
        
    Raises:
        ImageLoadError: If the image cannot be loaded
        InvalidConfigurationError: If invalid parameters are provided
    """
    
    def __init__(
        self,
        image_path: str,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        pooling_method: Literal["mean", "max", "min"] = DEFAULT_POOLING_METHOD,
        darkmode: bool = False,
    ) -> None:
        # Validate parameters
        if window_size <= 0:
            raise InvalidConfigurationError("Window size must be positive")
        if stride <= 0:
            raise InvalidConfigurationError("Stride must be positive")
        if pooling_method not in ["mean", "max", "min"]:
            raise InvalidConfigurationError(
                f"Invalid pooling method: {pooling_method}"
            )
            
        try:
            self.image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise ImageLoadError(image_path, FileNotFoundError(f"File not found: {image_path}"))
        except Exception as e:
            raise ImageLoadError(image_path, e)
            
        self.window_size = window_size
        self.stride = stride
        self.pooling_method = pooling_method
        self.darkmode = darkmode
        self.centroids: Optional[np.ndarray] = None

    def rgb_to_hex(self, rgb: Union[np.ndarray, Tuple[float, float, float]]) -> str:
        """
        Convert RGB values to hexadecimal color string.
        
        Args:
            rgb: RGB values as array or tuple
            
        Returns:
            Hexadecimal color string (e.g., '#FF0000')
        """
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0]), int(rgb[1]), int(rgb[2])
        )

    def get_brightness(self, rgb: Union[np.ndarray, Tuple[float, float, float]]) -> float:
        """
        Calculate brightness of an RGB color using luminance formula.
        
        Args:
            rgb: RGB values as array or tuple
            
        Returns:
            Brightness value (0-255)
        """
        return (
            BRIGHTNESS_WEIGHTS[0] * rgb[0]
            + BRIGHTNESS_WEIGHTS[1] * rgb[1]
            + BRIGHTNESS_WEIGHTS[2] * rgb[2]
        )

    def apply_pooling(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply pooling operation to reduce image dimensions.
        
        Args:
            arr: Input image array with shape (height, width, channels)
            
        Returns:
            Pooled image array with reduced dimensions
            
        Raises:
            InvalidConfigurationError: If pooling method is invalid
        """
        h, w, c = arr.shape
        new_h = (h - self.window_size) // self.stride + 1
        new_w = (w - self.window_size) // self.stride + 1
        
        if new_h <= 0 or new_w <= 0:
            raise InvalidConfigurationError(
                f"Pooling parameters result in invalid output size: "
                f"({new_h}, {new_w}). Adjust window_size and stride."
            )
            
        resized = np.zeros((new_h, new_w, c), dtype=arr.dtype)

        for i in range(new_h):
            for j in range(new_w):
                y_start = i * self.stride
                y_end = y_start + self.window_size
                x_start = j * self.stride
                x_end = x_start + self.window_size
                
                window = arr[y_start:y_end, x_start:x_end]

                if self.pooling_method == "mean":
                    resized[i, j] = window.mean(axis=(0, 1))
                elif self.pooling_method == "max":
                    resized[i, j] = window.max(axis=(0, 1))
                elif self.pooling_method == "min":
                    resized[i, j] = window.min(axis=(0, 1))
                else:
                    raise InvalidConfigurationError(
                        f"Unknown pooling method: {self.pooling_method}"
                    )
                    
        return resized

    def extract_palette(
        self,
        out_dir: str,
        pooling: bool = True,
        k: int = DEFAULT_K_COLORS,
        diversity: float = DEFAULT_DIVERSITY_FACTOR,
        output_json: Optional[str] = None,
        output_png: Optional[str] = None,
    ) -> Dict[str, Union[List[str], str]]:
        """
        Extract color palette from the image.
        
        Args:
            out_dir: Output directory for generated files
            pooling: Whether to apply pooling before clustering
            k: Number of colors to extract
            diversity: Diversity factor for color filtering (0-1)
            output_json: Name of output JSON file (optional)
            output_png: Name of output PNG file (optional)
            
        Returns:
            Dictionary containing extracted colors, background, and foreground
            
        Raises:
            InvalidConfigurationError: If parameters are invalid
        """
        if k <= 0:
            raise InvalidConfigurationError("Number of colors (k) must be positive")
        if not 0 <= diversity <= 1:
            raise InvalidConfigurationError("Diversity must be between 0 and 1")
            
        arr = np.array(self.image)
        
        if pooling:
            print("Pooling enabled.")
            print("Image size before pooling:", arr.shape)
            arr = self.apply_pooling(arr)
            print("Image size after pooling:", arr.shape)
        else:
            print("No pooling applied. Using original image size:", arr.shape)

        pixels = arr.reshape(-1, 3)
        k_ext = k * 2 if k < 10 else k
        
        km = KMeans(
            n_clusters=k_ext, 
            n_init=DEFAULT_K_MEANS_INIT, 
            random_state=DEFAULT_RANDOM_STATE
        )
        labels = km.fit_predict(pixels)
        centroids = km.cluster_centers_  # shape (k_ext, 3)

        self.centroids = centroids

        # Sort by frequency
        counts = np.bincount(labels)
        order = np.argsort(counts)[::-1]  # Most to least frequent
        centroids = centroids[order]
        counts = counts[order]

        # Find background and foreground colors
        brightness_values = np.array([self.get_brightness(c) for c in centroids])
        bg_idx = np.argmin(brightness_values)
        fg_idx = np.argmax(brightness_values)
        
        bg = centroids[bg_idx]
        fg = centroids[fg_idx]

        if self.darkmode:
            bg, fg = fg, bg

        # Remove bg, fg from centroids
        mask = np.ones(len(centroids), dtype=bool)
        mask[bg_idx] = False
        mask[fg_idx] = False
        centroids = centroids[mask]
        counts = counts[mask]

        # Apply diversity filtering
        min_dist = MIN_DIVERSITY_DISTANCE * diversity

        filtered_colors = []
        filtered_counts = []
        
        if len(centroids) > 0:
            filtered_colors = [centroids[0]]
            filtered_counts = [counts[0]]

            for i, color in enumerate(centroids[1:], start=1):
                distances = cdist([color], filtered_colors)
                if np.min(distances) > min_dist:
                    filtered_colors.append(color)
                    filtered_counts.append(counts[i])
                if len(filtered_colors) >= k:
                    break

            # If we don't have enough diverse colors, add remaining centroids
            while len(filtered_colors) < k and len(filtered_colors) < len(centroids):
                print("Not enough diverse colors found, adding more centroids.")
                remaining_idx = len(filtered_colors)
                if remaining_idx < len(centroids):
                    filtered_colors.append(centroids[remaining_idx])
                    filtered_counts.append(counts[remaining_idx])

        # Convert to hex
        colors_hex = [self.rgb_to_hex(c) for c in filtered_colors]
        bg_hex = self.rgb_to_hex(bg)
        fg_hex = self.rgb_to_hex(fg)

        data = {
            "colors": colors_hex,
            "background": bg_hex,
            "foreground": fg_hex,
        }

        # Save outputs
        if output_json is not None:
            self._save_json(data, out_dir, output_json)

        if output_png is not None:
            self._save_png(data, out_dir, output_png)

        return data

    def _save_json(
        self, 
        data: Dict[str, Union[List[str], str]], 
        out_dir: str, 
        filename: str
    ) -> None:
        """Save palette data to JSON file."""
        json_path = os.path.join(out_dir, filename)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Palette JSON saved to {json_path}")

    def _save_png(
        self, 
        data: Dict[str, Union[List[str], str]], 
        out_dir: str, 
        filename: str
    ) -> None:
        """Save palette visualization to PNG file."""
        # Type cast to ensure we have the correct types
        background = str(data["background"])
        foreground = str(data["foreground"])
        colors = list(data["colors"]) if isinstance(data["colors"], list) else []
        
        all_colors = [background, foreground] + colors
        all_labels = ["Background", "Foreground"] + [
            f"Color {i+1}" for i in range(len(colors))
        ]

        width = PALETTE_COLOR_WIDTH * len(all_colors)
        height = PALETTE_IMAGE_HEIGHT
        palette_img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(palette_img)
        font = ImageFont.load_default()

        for i, (color, label) in enumerate(zip(all_colors, all_labels)):
            x0 = i * PALETTE_COLOR_WIDTH
            x1 = (i + 1) * PALETTE_COLOR_WIDTH

            # Draw color rectangle
            draw.rectangle(
                [
                    x0 + PALETTE_MARGIN,
                    PALETTE_MARGIN,
                    x1 - PALETTE_MARGIN,
                    PALETTE_COLOR_RECT_HEIGHT + PALETTE_MARGIN,
                ],
                fill=color,
            )

            # Choose text color based on background brightness
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            brightness = self.get_brightness((r, g, b))
            text_color = (
                (255, 255, 255)
                if brightness < TEXT_COLOR_BRIGHTNESS_THRESHOLD
                else (0, 0, 0)
            )

            # Draw text labels
            text_x = x0 + 10
            draw.text(
                (text_x, PALETTE_TEXT_Y_POSITIONS[0]), 
                label, 
                fill=(0, 0, 0), 
                font=font
            )
            draw.text(
                (text_x, PALETTE_TEXT_Y_POSITIONS[1]), 
                color, 
                fill=(0, 0, 0), 
                font=font
            )
            draw.text(
                (text_x, PALETTE_TEXT_Y_POSITIONS[2]),
                f"RGB({r},{g},{b})",
                fill=(0, 0, 0),
                font=font,
            )

        png_path = os.path.join(out_dir, filename)
        palette_img.save(png_path)
        print(f"Saved PNG: {png_path}")