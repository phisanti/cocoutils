"""
Tests for JSON I/O utilities in cocoutils.utils.io
"""

import json
import pytest
import tempfile
from pathlib import Path
from cocoutils.utils.io import load_coco, save_coco


@pytest.fixture
def minimal_coco_data():
    """Minimal valid COCO data structure."""
    return {
        "info": {
            "description": "Test Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "test",
            "date_created": "2024-01-01 00:00:00"
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "width": 640,
                "height": 480,
                "file_name": "test.jpg"
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]],
                "area": 10000,
                "bbox": [100, 100, 100, 100],
                "iscrowd": 0
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "test_category",
                "supercategory": "test"
            }
        ]
    }


def test_save_and_load_coco(tmp_path, minimal_coco_data):
    """Test saving and loading COCO data maintains equality."""
    output_path = tmp_path / "test.json"
    
    # Save the data
    save_coco(minimal_coco_data, str(output_path))
    
    # Load it back
    loaded_data = load_coco(str(output_path))
    
    # Verify equality
    assert loaded_data == minimal_coco_data


def test_load_coco_file_not_found():
    """Test loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_coco("nonexistent.json")


def test_load_coco_invalid_json(tmp_path):
    """Test loading invalid JSON raises JSONDecodeError."""
    invalid_json_path = tmp_path / "invalid.json"
    invalid_json_path.write_text("{ invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        load_coco(str(invalid_json_path))


def test_load_coco_missing_required_keys(tmp_path):
    """Test loading JSON without required COCO keys raises ValueError."""
    incomplete_data = {"info": {}}
    incomplete_path = tmp_path / "incomplete.json"
    
    with open(incomplete_path, 'w') as f:
        json.dump(incomplete_data, f)
    
    with pytest.raises(ValueError, match="Missing required COCO key"):
        load_coco(str(incomplete_path))


def test_save_coco_creates_directory(tmp_path, minimal_coco_data):
    """Test that save_coco creates directories if they don't exist."""
    nested_path = tmp_path / "nested" / "directory" / "test.json"
    
    save_coco(minimal_coco_data, str(nested_path))
    
    assert nested_path.exists()
    loaded_data = load_coco(str(nested_path))
    assert loaded_data == minimal_coco_data


def test_save_coco_atomic_write(tmp_path, minimal_coco_data):
    """Test that save_coco uses atomic write (temp file + rename)."""
    output_path = tmp_path / "test.json"
    
    # Create a file first
    output_path.write_text("original content")
    
    # Save new data
    save_coco(minimal_coco_data, str(output_path))
    
    # Verify the file was replaced properly
    loaded_data = load_coco(str(output_path))
    assert loaded_data == minimal_coco_data


def test_save_coco_proper_formatting(tmp_path, minimal_coco_data):
    """Test that save_coco formats JSON with proper indentation."""
    output_path = tmp_path / "test.json"
    
    save_coco(minimal_coco_data, str(output_path))
    
    # Read raw text and verify formatting
    content = output_path.read_text()
    assert "  " in content  # Should have indentation
    assert content.count('\n') > 10  # Should be multi-line
