import numpy as np
import tempfile
import os
import json
from PIL import Image
from cocoutils.visualise.core import CocoVisualizer
import matplotlib.pyplot as plt

def create_dummy_coco(path, images, annotations, categories):
    with open(path, 'w') as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f, indent=2)

def test_visualizer_runs():
    with tempfile.TemporaryDirectory() as d:
        # 1. Create dummy image
        img_array = np.zeros((20, 20, 3), dtype=np.uint8)
        img_array[5:15, 5:15, 0] = 255  # Red square
        img = Image.fromarray(img_array)
        img_path = os.path.join(d, "test_img.png")
        img.save(img_path)

        # 2. Create dummy COCO file
        coco_path = os.path.join(d, "coco.json")
        images = [{"id": 1, "file_name": "test_img.png", "width": 20, "height": 20}]
        annotations = [{
            "id": 1, 
            "image_id": 1, 
            "category_id": 1, 
            "segmentation": [[5, 5, 15, 5, 15, 15, 5, 15]],
            "bbox": [5, 5, 10, 10],
            "area": 100,
            "iscrowd": 0
        }]
        categories = [{"id": 1, "name": "red_square"}]
        create_dummy_coco(coco_path, images, annotations, categories)

        # 3. Run the visualizer
        try:
            # Use a non-interactive backend for testing
            plt.switch_backend('Agg')
            
            visualizer = CocoVisualizer(coco_file=coco_path)
            
            fig, ax = plt.subplots()
            visualizer.visualize(image_path=img_path, ax=ax)
            
            # Check that something was drawn on the axes
            assert len(ax.patches) > 0, "No patches were drawn on the axes."
            assert len(ax.texts) > 0, "No text was drawn on the axes."

        except Exception as e:
            assert False, f"Visualizer raised an exception: {e}"
        finally:
            plt.close('all') # clean up figures
