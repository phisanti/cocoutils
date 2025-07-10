import numpy as np
import tempfile
import tifffile
import json
import os
from cocoutils.convert.core import CocoConverter
from cocoutils.reconstruct.core import CocoReconstructor

def test_simple_reconstruct():
    with tempfile.TemporaryDirectory() as d:
        # 1. Create a dummy mask and categories file
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1  # A 10x10 square of category 1
        original_mask_dir = os.path.join(d, "original_masks")
        os.makedirs(original_mask_dir)
        tifffile.imwrite(os.path.join(original_mask_dir, "test_mask.tif"), mask)

        cats = [{"id": 1, "name": "square"}]
        cats_path = os.path.join(d, "cats.json")
        with open(cats_path, "w") as f:
            json.dump(cats, f)

        # 2. Convert mask to COCO format
        coco_output_path = os.path.join(d, "coco.json")
        converter = CocoConverter(categories_path=cats_path)
        converter.from_masks(input_dir=original_mask_dir, output_file=coco_output_path)
        assert os.path.exists(coco_output_path)

        # 3. Reconstruct mask from COCO format
        reconstructed_mask_dir = os.path.join(d, "reconstructed_masks")
        reconstructor = CocoReconstructor()
        reconstructor.from_coco(coco_file=coco_output_path, output_dir=reconstructed_mask_dir, workers=1)
        
        reconstructed_mask_path = os.path.join(reconstructed_mask_dir, "test_mask.tif")
        assert os.path.exists(reconstructed_mask_path)

        # 4. Compare original and reconstructed masks
        rebuilt_mask = tifffile.imread(reconstructed_mask_path)
        
        intersection = np.logical_and(mask, rebuilt_mask).sum()
        union = np.logical_or(mask, rebuilt_mask).sum()
        iou = intersection / union if union > 0 else 1.0
        
        assert iou >= 0.95, f"IoU score {iou} is below the threshold of 0.95"
