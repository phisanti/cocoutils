"""
Test for the new visualize_annotations_masked function
"""

import numpy as np
import tempfile
import os
import json
from PIL import Image
from cocoutils.visualise import CocoVisualizer
import matplotlib.pyplot as plt


def create_coco(path, images, annotations, categories):
    """Helper to create dummy COCO file"""
    with open(path, 'w') as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f, indent=2)


def test_visualize_annotations_masked():
    """Test the new masked visualization function"""
    with tempfile.TemporaryDirectory() as d:
        # 1. Create dummy image
        img_array = np.zeros((20, 20, 3), dtype=np.uint8)
        img_array[5:15, 5:15, 0] = 255  # Red square
        img = Image.fromarray(img_array)
        img_path = os.path.join(d, "test_img.png")
        img.save(img_path)

        # 2. Create dummy COCO file with segmentation
        coco_path = os.path.join(d, "coco.json")
        images = [{"id": 1, "file_name": "test_img.png", "width": 20, "height": 20}]
        annotations = [{
            "id": 1, 
            "image_id": 1, 
            "category_id": 1, 
            "segmentation": [[5, 5, 15, 5, 15, 15, 5, 15]],  # Square segmentation
            "bbox": [5, 5, 10, 10],
            "area": 100,
            "iscrowd": 0
        }]
        categories = [{"id": 1, "name": "red_square"}]
        create_coco(coco_path, images, annotations, categories)

        # 3. Test the masked visualization
        try:
            # Use a non-interactive backend for testing
            plt.switch_backend('Agg')
            
            visualizer = CocoVisualizer(coco_file=coco_path)
            
            # Load the image as numpy array
            img_np = np.array(Image.open(img_path))
            
            # Test with show=False to avoid displaying in tests
            fig, ax = plt.subplots()
            result_ax = visualizer.visualize_annotations_masked(
                image=img_np, 
                annotation_ids=1, 
                ax=ax, 
                show=False
            )
            
            # Should return None when ax is provided
            assert result_ax is None, "Should return None when ax is provided"
            
            # Test without providing ax (should return ax)
            result_ax = visualizer.visualize_annotations_masked(
                image=img_np, 
                annotation_ids=[1], 
                show=False
            )
            
            assert result_ax is not None, "Should return axes when ax is not provided"
            
            print("visualize_annotations_masked tests passed!")

        except Exception as e:
            assert False, f"Masked visualizer raised an exception: {e}"
        finally:
            plt.close('all')  # clean up figures


def test_visualize_annotations_masked_no_annotations():
    """Test masked visualization with no annotations"""
    with tempfile.TemporaryDirectory() as d:
        # Create dummy image and empty COCO file
        img_array = np.zeros((20, 20, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = os.path.join(d, "test_img.png")
        img.save(img_path)

        coco_path = os.path.join(d, "coco.json")
        create_coco(coco_path, 
                         [{"id": 1, "file_name": "test_img.png", "width": 20, "height": 20}],
                         [], 
                         [])

        try:
            plt.switch_backend('Agg')
            visualizer = CocoVisualizer(coco_file=coco_path)
            img_np = np.array(Image.open(img_path))
            
            result_ax = visualizer.visualize_annotations_masked(
                image=img_np, 
                annotation_ids=999,  # Non-existent ID
                show=False
            )
            
            # Should return None when no annotations found
            assert result_ax is None, "Should return None when no annotations found"
            
            print("No annotations test passed!")

        except Exception as e:
            assert False, f"No annotations test raised an exception: {e}"
        finally:
            plt.close('all')


if __name__ == "__main__":
    test_visualize_annotations_masked()
    test_visualize_annotations_masked_no_annotations()
