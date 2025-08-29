import tempfile
import json
import os
from cocoutils.split import CocoSplitter


def test_simple_split():
    """Test basic COCO file splitting functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample COCO file with 2 images
        coco_data = {
            "info": {"description": "Test dataset"},
            "licenses": [],
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "image2.jpg", "width": 150, "height": 150}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0, "segmentation": []},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [30, 30, 15, 15], "area": 225, "iscrowd": 0, "segmentation": []},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 25, 25], "area": 625, "iscrowd": 0, "segmentation": []}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }
        
        # Save the sample COCO file
        input_file = os.path.join(temp_dir, "sample.json")
        with open(input_file, 'w') as f:
            json.dump(coco_data, f)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "split_output")
        
        # Split the file
        splitter = CocoSplitter()
        created_files = splitter.split_file(input_file, output_dir)
        
        # Verify results
        assert len(created_files) == 2
        
        # Check first split file (image1)
        image1_file = os.path.join(output_dir, "image1.json")
        assert image1_file in created_files
        
        with open(image1_file, 'r') as f:
            image1_coco = json.load(f)
        
        assert len(image1_coco["images"]) == 1
        assert image1_coco["images"][0]["file_name"] == "image1.jpg"
        assert len(image1_coco["annotations"]) == 2  # Two annotations for image1
        assert all(ann["image_id"] == 1 for ann in image1_coco["annotations"])
        
        # Check second split file (image2)
        image2_file = os.path.join(output_dir, "image2.json")
        assert image2_file in created_files
        
        with open(image2_file, 'r') as f:
            image2_coco = json.load(f)
        
        assert len(image2_coco["images"]) == 1
        assert image2_coco["images"][0]["file_name"] == "image2.jpg"
        assert len(image2_coco["annotations"]) == 1  # One annotation for image2
        assert image2_coco["annotations"][0]["image_id"] == 2


def test_split_by_categories():
    """Test splitting by category combinations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample COCO file
        coco_data = {
            "info": {"description": "Test dataset"},
            "licenses": [],
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "image2.jpg", "width": 150, "height": 150}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0, "segmentation": []},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [30, 30, 15, 15], "area": 225, "iscrowd": 0, "segmentation": []},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 25, 25], "area": 625, "iscrowd": 0, "segmentation": []}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }
        
        # Save the sample COCO file
        input_file = os.path.join(temp_dir, "sample.json")
        with open(input_file, 'w') as f:
            json.dump(coco_data, f)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "split_output")
        
        # Split by categories
        splitter = CocoSplitter()
        created_files = splitter.split_by_categories(input_file, output_dir)
        
        # Verify results
        assert len(created_files) == 2
        
        # Check that filenames contain category information
        filenames = [os.path.basename(f) for f in created_files]
        assert any("cat_dog" in name for name in filenames)  # image1 has both cats and dogs
        assert any("cat" in name for name in filenames)      # image2 has only cats


def test_custom_naming_pattern():
    """Test custom naming patterns for split files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple COCO file
        coco_data = {
            "info": {"description": "Test dataset"},
            "licenses": [],
            "images": [
                {"id": 42, "file_name": "test_image.png", "width": 100, "height": 100}
            ],
            "annotations": [
                {"id": 1, "image_id": 42, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0, "segmentation": []}
            ],
            "categories": [
                {"id": 1, "name": "object"}
            ]
        }
        
        # Save the sample COCO file
        input_file = os.path.join(temp_dir, "sample.json")
        with open(input_file, 'w') as f:
            json.dump(coco_data, f)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "split_output")
        
        # Split with custom naming pattern
        splitter = CocoSplitter()
        created_files = splitter.split_file(input_file, output_dir, "id_{image_id}_{image_name}")
        
        # Verify results
        assert len(created_files) == 1
        expected_filename = "id_42_test_image.json"
        assert os.path.basename(created_files[0]) == expected_filename


def test_empty_annotations():
    """Test splitting file with image that has no annotations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create COCO file with image but no annotations
        coco_data = {
            "info": {"description": "Test dataset"},
            "licenses": [],
            "images": [
                {"id": 1, "file_name": "empty_image.jpg", "width": 100, "height": 100}
            ],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "object"}
            ]
        }
        
        # Save the sample COCO file
        input_file = os.path.join(temp_dir, "sample.json")
        with open(input_file, 'w') as f:
            json.dump(coco_data, f)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "split_output")
        
        # Split the file
        splitter = CocoSplitter()
        created_files = splitter.split_file(input_file, output_dir)
        
        # Verify results
        assert len(created_files) == 1
        
        # Check the split file
        with open(created_files[0], 'r') as f:
            split_coco = json.load(f)
        
        assert len(split_coco["images"]) == 1
        assert len(split_coco["annotations"]) == 0  # No annotations
        assert split_coco["images"][0]["file_name"] == "empty_image.jpg"