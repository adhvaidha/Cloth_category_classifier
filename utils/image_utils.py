"""
Comprehensive Image Format Support Utilities

This module provides robust image loading and processing that supports
all major image formats including JPG, PNG, WEBP, GIF, BMP, TIFF, etc.
"""

import io
from typing import List, Optional, Tuple, Union
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError
import requests


# Enable PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported image formats
SUPPORTED_FORMATS = {
    # Raster formats
    'JPEG', 'JPG',  # JPEG
    'PNG',          # PNG
    'WEBP',         # WebP
    'GIF',          # GIF (static frames)
    'BMP',          # Bitmap
    'TIFF', 'TIF',  # TIFF
    'ICO',          # Icon
    'PCX',          # PC Paintbrush
    'PPM',          # Portable Pixmap
    'PBM',          # Portable Bitmap
    'PGM',          # Portable Graymap
    'XBM',          # X Bitmap
    'XPM',          # X Pixmap
    # Additional formats if available
    'HEIF', 'HEIC', # HEIF/HEIC (if pillow-heif installed)
    'AVIF',         # AVIF (if pillow-avif-plugin installed)
}

# File extensions mapping
EXTENSION_TO_FORMAT = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.png': 'PNG',
    '.webp': 'WEBP',
    '.gif': 'GIF',
    '.bmp': 'BMP',
    '.tiff': 'TIFF',
    '.tif': 'TIFF',
    '.ico': 'ICO',
    '.pcx': 'PCX',
    '.ppm': 'PPM',
    '.pbm': 'PBM',
    '.pgm': 'PGM',
    '.xbm': 'XBM',
    '.xpm': 'XPM',
    '.heif': 'HEIF',
    '.heic': 'HEIC',
    '.avif': 'AVIF',
}


def is_image_file(filepath: Union[str, Path]) -> bool:
    """
    Check if a file is a supported image format based on extension.
    
    Args:
        filepath: Path to the file
        
    Returns:
        True if the file appears to be a supported image format
    """
    path = Path(filepath)
    ext = path.suffix.lower()
    return ext in EXTENSION_TO_FORMAT


def get_image_format(filepath: Union[str, Path]) -> Optional[str]:
    """
    Get the image format from file extension.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Format name (e.g., 'JPEG', 'PNG') or None if unknown
    """
    path = Path(filepath)
    ext = path.suffix.lower()
    return EXTENSION_TO_FORMAT.get(ext)


def load_image_from_file(
    filepath: Union[str, Path],
    convert_to_rgb: bool = True,
    raise_on_error: bool = False
) -> Optional[Image.Image]:
    """
    Load an image from a file path, supporting all major formats.
    
    Enhanced to handle files without extensions (e.g., Gradio blob files)
    by detecting format from file content (magic bytes).
    
    Detection priority:
    1. File extension (if present)
    2. Magic byte detection (file content)
    3. PIL format detection (fallback)
    
    Args:
        filepath: Path to the image file
        convert_to_rgb: Convert image to RGB mode (required for models)
        raise_on_error: If True, raise exception on error; if False, return None
        
    Returns:
        PIL Image object or None if loading failed
    """
    try:
        path = Path(filepath)
        
        # Check if file exists
        if not path.exists():
            if raise_on_error:
                raise FileNotFoundError(f"Image file not found: {filepath}")
            return None
        
        # Step 1: Check file extension first
        detected_format = None
        if is_image_file(path):
            detected_format = get_image_format(path)
            if detected_format:
                print(f"🔍 Detected format from extension: {detected_format} (file: {path.name})")
        
        # Step 2: If no extension, detect from magic bytes (file content)
        # This is critical for Gradio blob files which have no extension
        if not detected_format:
            try:
                with open(path, 'rb') as f:
                    header = f.read(12)  # Read first 12 bytes for format detection
                    
                    # JPEG: FF D8 (JPEG files start with these bytes, third byte can vary)
                    if len(header) >= 2 and header[:2] == b'\xff\xd8':
                        detected_format = 'JPEG'
                    # PNG: 89 50 4E 47 0D 0A 1A 0A (PNG signature)
                    elif len(header) >= 8 and header[:8] == b'\x89PNG\r\n\x1a\n':
                        detected_format = 'PNG'
                    # GIF: 47 49 46 38 (GIF8 or GIF9)
                    elif len(header) >= 4 and header[:4] in [b'GIF8', b'GIF9']:
                        detected_format = 'GIF'
                    # WEBP: RIFF ... WEBP (RIFF container with WEBP format)
                    elif len(header) >= 12 and header[:4] == b'RIFF' and header[8:12] == b'WEBP':
                        detected_format = 'WEBP'
                    # BMP: 42 4D (BM - Windows bitmap)
                    elif len(header) >= 2 and header[:2] == b'BM':
                        detected_format = 'BMP'
                    # TIFF: 49 49 2A 00 (little-endian) or 4D 4D 00 2A (big-endian)
                    elif len(header) >= 4 and header[:4] in [b'II*\x00', b'MM\x00*']:
                        detected_format = 'TIFF'
                    
                    if detected_format:
                        print(f"🔍 Detected format from magic bytes: {detected_format} (file: {path.name})")
            except Exception as e:
                print(f"⚠️ Error reading file header for format detection: {e}")
        
        # Step 3: Try PIL format detection as fallback
        try:
            # Open image - PIL can also detect format from content
            test_img = Image.open(path)
            pil_format = test_img.format
            test_img.close()
            
            # Use PIL format if we didn't detect from extension/magic bytes, or verify consistency
            if pil_format:
                if detected_format:
                    # Both methods detected format - verify they match
                    if pil_format.upper() != detected_format.upper():
                        print(f"⚠️ Format mismatch: extension/magic bytes={detected_format}, PIL={pil_format}, using PIL format")
                    detected_format = pil_format  # Prefer PIL format as it's more reliable
                else:
                    # Only PIL detected format
                    detected_format = pil_format
                    print(f"🔍 Detected format from PIL: {detected_format} (file: {path.name})")
            elif not detected_format:
                # Neither method detected format
                if raise_on_error:
                    raise ValueError(f"Cannot identify image format and no valid extension: {filepath}")
                print(f"⚠️ Cannot identify image format: {filepath}")
                return None
            
            # Check if format is supported
            if detected_format and detected_format.upper() not in SUPPORTED_FORMATS:
                if raise_on_error:
                    raise ValueError(f"Unsupported image format: {detected_format} (file: {filepath})")
                print(f"⚠️ Skipping unsupported format: {detected_format} (file: {filepath})")
                return None
            
            # Verify it's actually a valid image
            with Image.open(path) as img:
                img.verify()
                
        except UnidentifiedImageError:
            # PIL can't identify it - but we might have detected from magic bytes
            if detected_format:
                # We detected format from magic bytes, try to open anyway
                print(f"⚠️ PIL couldn't identify format, but magic bytes suggest {detected_format}, attempting to open...")
                try:
                    # Try opening with PIL (it might still work)
                    test_img = Image.open(path)
                    test_img.verify()
                    test_img.close()
                except Exception as e:
                    if raise_on_error:
                        raise ValueError(f"File detected as {detected_format} but cannot be read: {filepath} ({str(e)})")
                    print(f"⚠️ File detected as {detected_format} but cannot be read: {filepath} ({str(e)})")
                    return None
            else:
                # No format detected from any method
                if raise_on_error:
                    raise ValueError(f"Cannot identify image format and no valid extension: {filepath}")
                print(f"⚠️ Cannot identify image format: {filepath}")
                return None
        
        # Re-open for actual use (verify() closes the file)
        img = Image.open(path)
        
        # Convert to RGB if needed (required for deep learning models)
        if convert_to_rgb:
            if img.mode != 'RGB':
                # Handle different modes
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
                else:
                    img = img.convert('RGB')
        
        return img
        
    except UnidentifiedImageError:
        error_msg = f"❌ Cannot identify image format: {filepath}"
        if raise_on_error:
            raise ValueError(error_msg)
        print(error_msg)
        return None
    except Exception as e:
        error_msg = f"❌ Error loading image {filepath}: {str(e)}"
        if raise_on_error:
            raise
        print(error_msg)
        return None


def load_image_from_bytes(
    image_bytes: bytes,
    convert_to_rgb: bool = True,
    raise_on_error: bool = False
) -> Optional[Image.Image]:
    """
    Load an image from bytes, supporting all major formats.
    
    Args:
        image_bytes: Image data as bytes
        convert_to_rgb: Convert image to RGB mode (required for models)
        raise_on_error: If True, raise exception on error; if False, return None
        
    Returns:
        PIL Image object or None if loading failed
    """
    try:
        # Open from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Verify it's actually an image
        img.verify()
        
        # Re-open for actual use
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if convert_to_rgb:
            if img.mode != 'RGB':
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
        
        return img
        
    except UnidentifiedImageError:
        error_msg = "❌ Cannot identify image format from bytes"
        if raise_on_error:
            raise ValueError(error_msg)
        print(error_msg)
        return None
    except Exception as e:
        error_msg = f"❌ Error loading image from bytes: {str(e)}"
        if raise_on_error:
            raise
        print(error_msg)
        return None


def load_image_from_url(
    url: str,
    timeout: int = 20,
    convert_to_rgb: bool = True,
    raise_on_error: bool = False
) -> Optional[Image.Image]:
    """
    Load an image from a URL, supporting all major formats.
    
    Args:
        url: URL to the image
        timeout: Request timeout in seconds
        convert_to_rgb: Convert image to RGB mode (required for models)
        raise_on_error: If True, raise exception on error; if False, return None
        
    Returns:
        PIL Image object or None if loading failed
    """
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        
        # Check content type
        content_type = resp.headers.get('Content-Type', '').lower()
        if not any(fmt in content_type for fmt in ['image', 'jpeg', 'png', 'webp', 'gif']):
            if raise_on_error:
                raise ValueError(f"URL does not point to an image: {url}")
            print(f"⚠️ URL does not appear to be an image: {url}")
            return None
        
        # Load from bytes
        return load_image_from_bytes(resp.content, convert_to_rgb, raise_on_error)
        
    except requests.RequestException as e:
        error_msg = f"❌ Error fetching image from URL {url}: {str(e)}"
        if raise_on_error:
            raise
        print(error_msg)
        return None
    except Exception as e:
        error_msg = f"❌ Error loading image from URL {url}: {str(e)}"
        if raise_on_error:
            raise
        print(error_msg)
        return None


def load_images_from_files(
    filepaths: List[Union[str, Path]],
    convert_to_rgb: bool = True,
    skip_errors: bool = True
) -> List[Image.Image]:
    """
    Load multiple images from file paths, supporting all major formats.
    
    Args:
        filepaths: List of paths to image files
        convert_to_rgb: Convert images to RGB mode (required for models)
        skip_errors: If True, skip files that fail to load; if False, raise on first error
        
    Returns:
        List of PIL Image objects (only successfully loaded images)
    """
    images = []
    loaded_count = 0
    failed_count = 0
    
    for fp in filepaths:
        img = load_image_from_file(fp, convert_to_rgb, raise_on_error=not skip_errors)
        if img is not None:
            images.append(img)
            loaded_count += 1
        else:
            failed_count += 1
    
    if failed_count > 0:
        print(f"⚠️ Loaded {loaded_count} images, {failed_count} failed")
    
    return images


def validate_image_format(img: Image.Image) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image is in a supported format and ready for processing.
    
    Args:
        img: PIL Image object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if img is None:
        return False, "Image is None"
    
    if not hasattr(img, 'mode'):
        return False, "Invalid image object"
    
    # Check if format is supported
    if hasattr(img, 'format') and img.format:
        if img.format not in SUPPORTED_FORMATS:
            return False, f"Unsupported format: {img.format}"
    
    # Check if image has valid size
    if img.size[0] == 0 or img.size[1] == 0:
        return False, "Image has zero dimensions"
    
    return True, None


def ensure_rgb_image(img: Image.Image) -> Image.Image:
    """
    Ensure an image is in RGB mode, converting if necessary.
    
    Args:
        img: PIL Image object
        
    Returns:
        RGB mode PIL Image
    """
    if img.mode == 'RGB':
        return img
    
    if img.mode in ('RGBA', 'LA', 'P'):
        # Handle transparency
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        if img.mode in ('RGBA', 'LA'):
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[-1])
            else:
                background.paste(img, mask=img.split()[-1])
        return background
    else:
        return img.convert('RGB')


def get_supported_formats() -> List[str]:
    """
    Get list of all supported image formats.
    
    Returns:
        List of format names
    """
    return sorted(list(SUPPORTED_FORMATS))


def get_supported_extensions() -> List[str]:
    """
    Get list of all supported file extensions.
    
    Returns:
        List of file extensions (with dots)
    """
    return sorted(list(EXTENSION_TO_FORMAT.keys()))
