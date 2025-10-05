"""
Custom exception classes for the palette extractor package.
"""

from typing import Optional


class PaletteExtractorError(Exception):
    """Base exception class for all palette extractor errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.cause = cause


class ImageLoadError(PaletteExtractorError):
    """Raised when an image cannot be loaded or processed."""
    
    def __init__(self, image_path: str, cause: Optional[Exception] = None) -> None:
        message = f"Failed to load image: {image_path}"
        if cause:
            message += f" (Reason: {str(cause)})"
        super().__init__(message, cause)
        self.image_path = image_path


class InvalidConfigurationError(PaletteExtractorError):
    """Raised when invalid configuration parameters are provided."""
    pass


class InsufficientColorsError(PaletteExtractorError):
    """Raised when not enough diverse colors can be extracted from an image."""
    pass