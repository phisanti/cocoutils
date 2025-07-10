from typing import Dict, Any, List, Tuple
from ..utils.io import load_coco, save_coco

class CocoMerger:
    """
    Merges two COCO annotation files.
    """

    def __init__(self):
        pass

    def merge_files(self, file1_path: str, file2_path: str, output_path: str):
        """
        Merges two COCO files into a single file.

        Args:
            file1_path (str): Path to the first COCO file.
            file2_path (str): Path to the second COCO file.
            output_path (str): Path to save the merged COCO file.
        """
        coco1 = load_coco(file1_path)
        coco2 = load_coco(file2_path)

        # Basic validation
        self._validate_categories(coco1.get('categories', []), coco2.get('categories', []))

        # Start with a copy of the first COCO file
        merged_coco = coco1.copy()

        # Remap IDs to avoid collisions
        img_id_map, ann_id_map = self._create_id_maps(coco1, coco2)

        # Merge images
        for img in coco2.get('images', []):
            original_id = img['id']
            img['id'] = img_id_map[original_id]
            merged_coco['images'].append(img)

        # Merge annotations
        for ann in coco2.get('annotations', []):
            original_img_id = ann['image_id']
            original_ann_id = ann['id']
            ann['image_id'] = img_id_map[original_img_id]
            ann['id'] = ann_id_map[original_ann_id]
            merged_coco['annotations'].append(ann)
            
        # For simplicity, we'll use the categories from the second file,
        # as they have been validated to be compatible.
        merged_coco['categories'] = coco2.get('categories', [])

        save_coco(merged_coco, output_path)

    def _validate_categories(self, cats1: List[Dict[str, Any]], cats2: List[Dict[str, Any]]):
        """
        Validates that the categories are compatible between the two files.
        Raises an assertion error if they are not.
        """
        cats1_map = {c['name']: c['id'] for c in cats1}
        cats2_map = {c['name']: c['id'] for c in cats2}

        if set(cats1_map.keys()) != set(cats2_map.keys()):
            raise ValueError("Category names do not match between files.")

        for name, id1 in cats1_map.items():
            if id1 != cats2_map[name]:
                raise ValueError(f"Category '{name}' has mismatched IDs: {id1} and {cats2_map[name]}.")

    def _create_id_maps(self, coco1: Dict[str, Any], coco2: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Creates mapping for image and annotation IDs to avoid collisions.
        """
        max_img_id1 = max([img['id'] for img in coco1.get('images', [])] or [0])
        max_ann_id1 = max([ann['id'] for ann in coco1.get('annotations', [])] or [0])

        img_id_map = {img['id']: img['id'] + max_img_id1 + 1 for img in coco2.get('images', [])}
        ann_id_map = {ann['id']: ann['id'] + max_ann_id1 + 1 for ann in coco2.get('annotations', [])}
        
        return img_id_map, ann_id_map
