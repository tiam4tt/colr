# Colr

A Python package for extracting color palettes from images using 
K-means clustering with optional image pooling and color diversity filtering.

## Features

- Extract dominant colors from any image
- Optional image pooling for performance optimization
- Color diversity filtering to avoid similar colors
- Support for dark/light mode palette generation
- CLI tool and Python API
- Export results to JSON and PNG formats

## Installation

```bash
pip install colr
```

Or install from source:

```bash
git clone https://github.com/tiam4tt/colr.git
cd colr
pip install -e .
```

## Command Line Usage

```bash
# Basic usage
colr image.jpg

# Extract 8 colors with pooling enabled
colr image.jpg --pooling -k 8

# Custom output directory and filenames
colr image.jpg --out-dir ./output --output-json colors.json --output-png palette.png

# Dark mode (swap background/foreground)
colr image.jpg --darkmode

# High diversity filtering
colr image.jpg --diversity 0.8
```

## Python API Usage

```python
from colr import ColorPaletteExtractor

# Create extractor
extractor = ColorPaletteExtractor("image.jpg")

# Extract palette
palette = extractor.extract_palette(
    out_dir="./output",
    pooling=True,
    k=6,
    diversity=0.4,
    output_json="palette.json",
    output_png="palette.png"
)

print(f"Background: {palette['background']}")
print(f"Foreground: {palette['foreground']}")
print(f"Colors: {palette['colors']}")
```

## Parameters

- `k`: Number of colors to extract (default: 6)
- `diversity`: Color diversity factor 0-1 (default: 0.4)
- `pooling`: Enable image pooling for performance (default: False)
- `window_size`: Pooling window size (default: 4)
- `stride`: Pooling stride (default: 4)
- `pooling_method`: Pooling method - 'mean', 'max', or 'min' (default: 'mean')
- `darkmode`: Swap background/foreground colors (default: False)

## Output Format

The palette data is returned as a dictionary:

```python
{
    "background": "#1a1a1a",     # Darkest color
    "foreground": "#ffffff",     # Lightest color  
    "colors": [                  # Extracted palette colors
        "#ff5733",
        "#33ff57",
        "#3357ff",
        ...
    ]
}
```

## Requirements

- Python 3.8+
- NumPy
- Pillow (PIL)
- scikit-learn
- SciPy

## License

MIT License - see LICENSE file for details.