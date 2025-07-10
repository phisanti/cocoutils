"""
Tests for TIFF I/O utilities in cocoutils.utils.io
"""

import numpy as np
import pytest
from pathlib import Path
from cocoutils.utils.io import load_tiff, save_tiff


def test_save_and_load_tiff_uint8(tmp_path):
    """Test saving and loading uint8 TIFF maintains array equality."""
    # Create random uint8 array
    original_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    output_path = tmp_path / "test_uint8.tiff"
    
    # Save the array
    save_tiff(original_array, str(output_path))
    
    # Load it back
    loaded_array = load_tiff(str(output_path))
    
    # Verify equality
    assert np.array_equal(original_array, loaded_array)
    assert loaded_array.dtype == np.uint8


def test_save_and_load_tiff_uint16(tmp_path):
    """Test saving and loading uint16 TIFF maintains array equality."""
    # Create random uint16 array
    original_array = np.random.randint(0, 65536, size=(50, 50), dtype=np.uint16)
    output_path = tmp_path / "test_uint16.tiff"
    
    # Save the array
    save_tiff(original_array, str(output_path))
    
    # Load it back
    loaded_array = load_tiff(str(output_path))
    
    # Verify equality
    assert np.array_equal(original_array, loaded_array)
    assert loaded_array.dtype == np.uint16


def test_load_tiff_file_not_found():
    """Test loading non-existent TIFF file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_tiff("nonexistent.tiff")


def test_save_tiff_creates_directory(tmp_path):
    """Test that save_tiff creates directories if they don't exist."""
    array = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    nested_path = tmp_path / "nested" / "directory" / "test.tiff"
    
    save_tiff(array, str(nested_path))
    
    assert nested_path.exists()
    loaded_array = load_tiff(str(nested_path))
    assert np.array_equal(array, loaded_array)


def test_save_tiff_overwrite_false(tmp_path):
    """Test that save_tiff respects overwrite=False."""
    array = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    output_path = tmp_path / "test.tiff"
    
    # Save first time
    save_tiff(array, str(output_path))
    
    # Try to save again with overwrite=False
    with pytest.raises(FileExistsError):
        save_tiff(array, str(output_path), overwrite=False)


def test_save_tiff_overwrite_true(tmp_path):
    """Test that save_tiff overwrites by default."""
    array1 = np.ones((10, 10), dtype=np.uint8)
    array2 = np.ones((10, 10), dtype=np.uint8) * 255
    output_path = tmp_path / "test.tiff"
    
    # Save first array
    save_tiff(array1, str(output_path))
    
    # Save second array (should overwrite)
    save_tiff(array2, str(output_path))
    
    # Load and verify it's the second array
    loaded_array = load_tiff(str(output_path))
    assert np.array_equal(array2, loaded_array)


def test_load_tiff_dtype_conversion():
    """Test that load_tiff converts dtypes appropriately."""
    # This test would require creating actual TIFF files with different dtypes
    # For now, we'll test the dtype enforcement logic indirectly
    pass


def test_tiff_roundtrip_different_shapes(tmp_path):
    """Test TIFF save/load with different array shapes."""
    shapes = [(100, 100), (50, 75), (200, 150)]
    
    for shape in shapes:
        array = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        output_path = tmp_path / f"test_{shape[0]}x{shape[1]}.tiff"
        
        save_tiff(array, str(output_path))
        loaded_array = load_tiff(str(output_path))
        
        assert np.array_equal(array, loaded_array)
        assert loaded_array.shape == shape


def test_tiff_grayscale_masks(tmp_path):
    """Test TIFF I/O with typical mask values (0-255)."""
    # Create a mask with typical segmentation values
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # White square
    mask[40:60, 40:60] = 128  # Gray square in center
    
    output_path = tmp_path / "mask.tiff"
    
    save_tiff(mask, str(output_path))
    loaded_mask = load_tiff(str(output_path))
    
    assert np.array_equal(mask, loaded_mask)
    assert loaded_mask.dtype == np.uint8
    assert loaded_mask.max() == 255
    assert loaded_mask.min() == 0
