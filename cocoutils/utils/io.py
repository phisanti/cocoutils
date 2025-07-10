"""
I/O utilities for COCO format JSON files and TIFF images.

This module provides centralized I/O functions with validation and error handling
for the cocoutils package.
"""

import json
import os
import tempfile
import numpy as np
import tifffile
from pathlib import Path
from typing import Dict, Any
import warnings


def load_coco(path: str) -> Dict[str, Any]:
    """
    Load COCO format annotations from a JSON file.
    
    Args:
        path (str): Path to the COCO JSON file.
        
    Returns:
        Dict[str, Any]: COCO data structure.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the file doesn't contain required COCO keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"COCO file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            coco_data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {path}: {e.msg}", e.doc, e.pos)
    
    # Validate basic COCO schema keys
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in coco_data:
            raise ValueError(f"Missing required COCO key '{key}' in file {path}")
    
    return coco_data


def save_coco(obj: Dict[str, Any], path: str) -> None:
    """
    Save COCO data structure to a JSON file with atomic write.
    
    Args:
        obj (Dict[str, Any]): COCO data structure to save.
        path (str): Output file path.
        
    Raises:
        OSError: If directory creation or file writing fails.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Atomic write using temporary file
    path_obj = Path(path)
    temp_path = path_obj.parent / f".tmp_{path_obj.name}"
    
    try:
        with open(temp_path, 'w') as f:
            json.dump(obj, f, indent=2)
        
        # Atomic rename
        os.rename(temp_path, path)
    except Exception as e:
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise OSError(f"Failed to save COCO file {path}: {e}")


def load_tiff(path: str) -> np.ndarray:
    """
    Load TIFF image with validation.
    
    Args:
        path (str): Path to the TIFF file.
        
    Returns:
        np.ndarray: Image array.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file cannot be read or has invalid format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"TIFF file not found: {path}")
    
    try:
        img = tifffile.imread(path)
    except Exception as e:
        raise ValueError(f"Failed to read TIFF file {path}: {e}")
    
    # Ensure proper dtype
    if img.dtype not in [np.uint8, np.uint16]:
        # Convert to uint8 if possible, otherwise uint16
        if img.max() <= 255:
            warnings.warn(
                f"Image dtype {img.dtype} is not uint8/uint16. Forcing conversion to uint8."
            )
            img = img.astype(np.uint8)
        else:
            warnings.warn(
                f"Image dtype {img.dtype} is not uint8/uint16. Forcing conversion to uint16."
            )
            img = img.astype(np.uint16)
    
    return img


def save_tiff(arr: np.ndarray, path: str, overwrite: bool = True) -> None:
    """
    Save numpy array as TIFF image.
    
    Args:
        arr (np.ndarray): Image array to save.
        path (str): Output file path.
        overwrite (bool): Whether to overwrite existing files.
        
    Raises:
        FileExistsError: If file exists and overwrite is False.
        OSError: If directory creation or file writing fails.
    """
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File exists and overwrite=False: {path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        tifffile.imwrite(path, arr)
    except Exception as e:
        raise OSError(f"Failed to save TIFF file {path}: {e}")
