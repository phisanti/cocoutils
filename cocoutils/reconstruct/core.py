import os
import json
import numpy as np
import tifffile
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List

from ..utils.geometry import create_segmentation_mask

class CocoReconstructor:
    """
    Reconstructs segmentation masks from COCO format annotations.
    """

    def __init__(self):
        pass

    def from_coco(self, coco_file: str, output_dir: str, workers: int = 0):
        """
        Reconstructs masks from a COCO file and saves them to a directory.

        Args:
            coco_file (str): Path to the COCO annotations JSON file.
            output_dir (str): Directory to save the generated mask images.
            workers (int): Number of parallel workers (0 = all cores, 1 = sequential).
        """
        os.makedirs(output_dir, exist_ok=True)
        coco_data = self._load_coco_annotations(coco_file)
        
        if workers is None or workers == 1:
            for img_info in tqdm(coco_data['images'], desc="Creating masks (sequential)"):
                self._process_single_image(img_info, coco_data, output_dir)
        else:
            num_workers = multiprocessing.cpu_count() if workers == 0 else workers
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(self._process_single_image, img, coco_data, output_dir)
                    for img in coco_data['images']
                ]
                for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Creating masks ({num_workers} workers)"):
                    pass

    def _load_coco_annotations(self, coco_file: str) -> Dict[str, Any]:
        """Loads COCO annotations from a JSON file."""
        with open(coco_file, 'r') as f:
            return json.load(f)

    def _create_mask_from_annotations(self, coco_data: Dict[str, Any], image_id: int, width: int, height: int) -> np.ndarray:
        """Creates a mask for a single image."""
        mask = np.zeros((height, width), dtype=np.uint8)
        annotations = [ann for ann in coco_data.get("annotations", []) if ann.get("image_id") == image_id]
        
        for ann in annotations:
            category_id = ann.get("category_id")
            segmentation = ann.get("segmentation")
            
            # The new converter does not produce segmentation_types, so we can't rely on it.
            # We will determine orientation on the fly.
            if not segmentation or not category_id:
                continue

            # create_segmentation_mask can now determine orientation if types are None
            full_mask_tensor = create_segmentation_mask(segmentation, None, height, width)
            
            if full_mask_tensor is not None:
                binary_mask = full_mask_tensor.numpy().astype(np.uint8)
                # Place the object on the main mask, respecting already placed objects
                mask = np.where(binary_mask > 0, category_id, mask)
                
        return mask

    def _process_single_image(self, img_info: Dict[str, Any], coco_data: Dict[str, Any], output_dir: str):
        """Processes one image: creates and saves the mask."""
        image_id = img_info["id"]
        width = img_info["width"]
        height = img_info["height"]
        file_name = img_info["file_name"]
        
        mask = self._create_mask_from_annotations(coco_data, image_id, width, height)
        
        out_path = os.path.join(output_dir, file_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tifffile.imwrite(out_path, mask)
