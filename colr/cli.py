"""
Command-line interface for the palette extractor.

This module provides the main CLI entry point for the palette-extractor package.
"""

import argparse
import os
import sys
from typing import Optional

from .config import (
    DEFAULT_DIVERSITY_FACTOR,
    DEFAULT_JSON_FILENAME,
    DEFAULT_K_COLORS,
    DEFAULT_PNG_FILENAME,
    DEFAULT_POOLING_METHOD,
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    POOLING_METHODS,
)
from .exceptions import PaletteExtractorError
from .extractor import ColorPaletteExtractor


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract color palette from an image using K-means clustering.",
        prog="colr",
    )
    
    # Required arguments
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file",
    )
    
    # Optional arguments
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory for JSON and PNG files (default: current directory)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Window size for pooling (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Stride for pooling (default: {DEFAULT_STRIDE})",
    )
    parser.add_argument(
        "--pooling-method",
        type=str,
        choices=POOLING_METHODS,
        default=DEFAULT_POOLING_METHOD,
        help=f"Pooling method (default: {DEFAULT_POOLING_METHOD})",
    )
    parser.add_argument(
        "--pooling",
        action="store_true",
        help="Enable pooling before clustering",
    )
    parser.add_argument(
        "-k",
        "--colors",
        type=int,
        default=DEFAULT_K_COLORS,
        help=f"Number of colors to extract (default: {DEFAULT_K_COLORS})",
    )
    parser.add_argument(
        "--diversity",
        type=float,
        default=DEFAULT_DIVERSITY_FACTOR,
        help=f"Diversity factor for color filtering, 0-1 (default: {DEFAULT_DIVERSITY_FACTOR})",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_JSON_FILENAME,
        help=f"Output JSON file name (default: {DEFAULT_JSON_FILENAME})",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=DEFAULT_PNG_FILENAME,
        help=f"Output PNG file name (default: {DEFAULT_PNG_FILENAME})",
    )
    parser.add_argument(
        "--darkmode",
        action="store_true",
        help="Enable dark mode (swap background/foreground colors)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON output generation",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG output generation",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Raises:
        SystemExit: If validation fails
    """
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)
        
    if args.window_size <= 0:
        print("Error: Window size must be positive", file=sys.stderr)
        sys.exit(1)
        
    if args.stride <= 0:
        print("Error: Stride must be positive", file=sys.stderr)
        sys.exit(1)
        
    if args.colors <= 0:
        print("Error: Number of colors must be positive", file=sys.stderr)
        sys.exit(1)
        
    if not 0 <= args.diversity <= 1:
        print("Error: Diversity must be between 0 and 1", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_args(args)
    
    try:
        # Create the extractor
        extractor = ColorPaletteExtractor(
            image_path=args.image_path,
            window_size=args.window_size,
            stride=args.stride,
            pooling_method=args.pooling_method,
            darkmode=args.darkmode,
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(args.out_dir, exist_ok=True)
        
        # Determine output files
        output_json: Optional[str] = None if args.no_json else args.output_json
        output_png: Optional[str] = None if args.no_png else args.output_png
        
        # Extract palette
        palette = extractor.extract_palette(
            out_dir=args.out_dir,
            pooling=args.pooling,
            k=args.colors,
            diversity=args.diversity,
            output_json=output_json,
            output_png=output_png,
        )
        
        # Print summary
        print(f"Extracted {len(palette['colors'])} colors:")
        print(f"Background: {palette['background']}")
        print(f"Foreground: {palette['foreground']}")
        print("Colors:", ", ".join(palette["colors"]))
        
        if not args.no_json and not args.no_png:
            print(f"Output files saved to: {args.out_dir}")
        elif not args.no_json:
            print(f"JSON file saved to: {args.out_dir}")
        elif not args.no_png:
            print(f"PNG file saved to: {args.out_dir}")
            
    except PaletteExtractorError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Operation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()