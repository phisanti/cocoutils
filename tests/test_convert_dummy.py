import numpy as np
import tempfile
import tifffile
import json
import os
from cocoutils.convert.core import CocoConverter
from cocoutils.utils.categories import CategoryManager

def test_simple_convert():
    with tempfile.TemporaryDirectory() as d:
        # Create a dummy mask
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:5, 1:5] = 1
        mask_path = os.path.join(d, "m.tif")
        tifffile.imwrite(mask_path, mask)

        # Create a dummy categories file
        cats = [{"id": 1, "name": "obj"}]
        cats_path = os.path.join(d, "cats.json")
        with open(cats_path, "w") as f:
            json.dump(cats, f)

        # Run the converter
        output_json_path = os.path.join(d, "out.json")
        converter = CocoConverter(categories_path=cats_path)
        converter.from_masks(input_dir=d, output_file=output_json_path)

        # Check that the output file exists
        assert os.path.exists(output_json_path)

        # Check the content of the output file
        with open(output_json_path, 'r') as f:
            coco_data = json.load(f)
        
        assert len(coco_data['images']) == 1
        assert len(coco_data['annotations']) == 1
        assert coco_data['annotations'][0]['category_id'] == 1
        assert coco_data['categories'][0]['name'] == 'obj'
