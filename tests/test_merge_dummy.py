import json
import tempfile
import os
from cocoutils.merge.core import CocoMerger

def create_dummy_coco(path, images, annotations, categories):
    with open(path, 'w') as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f, indent=2)

def test_simple_merge():
    with tempfile.TemporaryDirectory() as d:
        # Define categories, must be identical
        cats = [{"id": 1, "name": "obj"}]

        # Create first dummy COCO file
        coco1_path = os.path.join(d, "coco1.json")
        images1 = [{"id": 1, "file_name": "img1.tif", "width": 10, "height": 10}]
        annotations1 = [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0,0,5,5,0,5]]}]
        create_dummy_coco(coco1_path, images1, annotations1, cats)

        # Create second dummy COCO file
        coco2_path = os.path.join(d, "coco2.json")
        images2 = [{"id": 1, "file_name": "img2.tif", "width": 10, "height": 10}]
        annotations2 = [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[1,1,6,6,1,6]]}]
        create_dummy_coco(coco2_path, images2, annotations2, cats)

        # Merge the files
        merged_path = os.path.join(d, "merged.json")
        merger = CocoMerger()
        merger.merge_files(coco1_path, coco2_path, merged_path)

        # Verify the merged file
        assert os.path.exists(merged_path)
        with open(merged_path, 'r') as f:
            merged_data = json.load(f)

        assert len(merged_data["images"]) == 2
        assert len(merged_data["annotations"]) == 2
        assert len(merged_data["categories"]) == 1
        
        # Check that IDs have been remapped and are unique
        img_ids = {img['id'] for img in merged_data['images']}
        ann_ids = {ann['id'] for ann in merged_data['annotations']}
        assert len(img_ids) == 2
        assert len(ann_ids) == 2
        
        # Check that image_id in annotations was updated correctly
        ann2 = next(ann for ann in merged_data['annotations'] if ann['id'] > 1)
        img2 = next(img for img in merged_data['images'] if img['file_name'] == 'img2.tif')
        assert ann2['image_id'] == img2['id']

def test_merge_with_category_mismatch():
    with tempfile.TemporaryDirectory() as d:
        cats1 = [{"id": 1, "name": "obj1"}]
        cats2 = [{"id": 1, "name": "obj2"}]
        
        coco1_path = os.path.join(d, "coco1.json")
        create_dummy_coco(coco1_path, [], [], cats1)
        
        coco2_path = os.path.join(d, "coco2.json")
        create_dummy_coco(coco2_path, [], [], cats2)
        
        merger = CocoMerger()
        try:
            merger.merge_files(coco1_path, coco2_path, os.path.join(d, "merged.json"))
            assert False, "Should have raised ValueError for category mismatch"
        except ValueError as e:
            assert "Category names do not match" in str(e)

def test_merge_with_id_mismatch():
    with tempfile.TemporaryDirectory() as d:
        cats1 = [{"id": 1, "name": "obj"}]
        cats2 = [{"id": 2, "name": "obj"}]
        
        coco1_path = os.path.join(d, "coco1.json")
        create_dummy_coco(coco1_path, [], [], cats1)
        
        coco2_path = os.path.join(d, "coco2.json")
        create_dummy_coco(coco2_path, [], [], cats2)
        
        merger = CocoMerger()
        try:
            merger.merge_files(coco1_path, coco2_path, os.path.join(d, "merged.json"))
            assert False, "Should have raised ValueError for ID mismatch"
        except ValueError as e:
            assert "has mismatched IDs" in str(e)
