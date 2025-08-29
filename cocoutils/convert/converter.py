import os
import numpy as np
import datetime
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import label
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from typing import List, Dict, Any, Tuple

from ..utils.categories import CategoryManager
from ..utils.geometry import create_segmentation_mask, bbox_from_polygons
from ..utils.io import save_coco, load_tiff

class CocoConverter:
    """
    Converts segmentation masks to COCO format.
    """

    def __init__(self, categories_path: str):
        """
        Initializes the CocoConverter.

        Args:
            categories_path (str): Path to the categories JSON file.
        """
        self.category_manager = CategoryManager(categories_path)
        self.coco_data = self._initialize_coco_structure()

    def from_masks(self, input_dir: str, output_file: str, per_file: bool = False):
        """
        Converts segmentation masks from a directory to COCO JSON file(s).

        Args:
            input_dir (str): Path to the directory containing TIFF masks.
            output_file (str): Path to save the output COCO JSON file.
            per_file (bool): If True, create a separate COCO JSON file for each image.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        tiff_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))])
        if not tiff_files:
            print(f"No TIFF files found in {input_dir}")
            return

        if per_file:
            self._process_per_file(input_dir, output_file, tiff_files)
            print(f"Processed {len(tiff_files)} images (per-file mode).")
        else:
            self._process_single_file(input_dir, output_file, tiff_files)

    def _process_single_file(self, input_dir: str, output_file: str, tiff_files: List[str]):
        """
        Processes all TIFF files and writes a single consolidated COCO JSON file.

        Args:
            input_dir (str): Directory containing TIFF mask images.
            output_file (str): Output COCO JSON file.
            tiff_files (List[str]): List of TIFF filenames to process.
        
        Creates files named: {output_stem}_{image_stem}.json
        """
        annotation_id = 1
        for image_id, tiff_file in enumerate(tqdm(tiff_files, desc="Processing images"), 1):
            file_path = os.path.join(input_dir, tiff_file)
            try:
                img = load_tiff(file_path)
                image_info, coco_annotations = self._process_image(img, image_id, tiff_file, annotation_id)
                if image_info:
                    self.coco_data["images"].append(image_info)
                if coco_annotations:
                    self.coco_data["annotations"].extend(coco_annotations)
                    annotation_id += len(coco_annotations)
            except Exception as e:
                print(f"Error processing {tiff_file}: {e}")
        self._save_annotations(output_file)
        print(f"Processed {len(tiff_files)} images with {annotation_id - 1} annotations.")
        print(f"COCO annotations saved to {output_file}")
    def _process_per_file(self, input_dir: str, output_file: str, tiff_files: List[str]):
        """
        Processes each TIFF file in input_dir and writes a separate COCO JSON file for each image.

        Args:
            input_dir (str): Directory containing TIFF mask images.
            output_file (str): Output file pattern (used for stem).
            tiff_files (List[str]): List of TIFF filenames to process.
        """
        for tiff_file in tqdm(tiff_files, desc="Processing images (per-file)"):
            file_path = os.path.join(input_dir, tiff_file)
            try:
                img = load_tiff(file_path)
                self.coco_data = self._initialize_coco_structure()
                image_info, coco_annotations = self._process_image(img, 1, tiff_file, 1)
                if image_info:
                    self.coco_data["images"].append(image_info)
                if coco_annotations:
                    self.coco_data["annotations"].extend(coco_annotations)
                stem = Path(output_file).stem
                out_file = os.path.join(
                    os.path.dirname(output_file),
                    f"{stem}_{Path(tiff_file).stem}.json"
                )
                self._save_annotations(out_file)
                print(f"COCO annotations saved to {out_file}")
            except Exception as e:
                print(f"Error processing {tiff_file}: {e}")


    def _initialize_coco_structure(self) -> Dict[str, Any]:
        """
        Initializes the basic COCO data structure.
        """
        return {
            "info": {
                "description": "Converted Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "cocoutils",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": self.category_manager.categories
        }

    def _process_image(self, img: np.ndarray, image_id: int, tiff_file: str, start_annotation_id: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Processes a single image to extract COCO annotations.
        """
        coco_annotations = []
        height, width = img.shape
        image_info = {
            "id": image_id,
            "file_name": tiff_file,
            "width": width,
            "height": height,
            "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "license": None
        }
        
        class_ids = np.unique(img)
        class_ids = class_ids[class_ids > 0]
        
        annotation_id = start_annotation_id
        for class_id in class_ids:
            if class_id not in self.category_manager.id_to_name:
                continue

            class_mask = (img == class_id)
            labeled_mask, num_objects = measure.label(class_mask, connectivity=1, return_num=True)
            
            for obj_idx in range(1, num_objects + 1):
                sub_mask = (labeled_mask == obj_idx).astype(np.uint8)
                if np.sum(sub_mask) < 10:  # Skip tiny objects
                    continue
                
                annotation = self._create_sub_mask_annotation(sub_mask, image_id, int(class_id), annotation_id)
                if annotation:
                    coco_annotations.append(annotation)
                    annotation_id += 1
                    
        return image_info, coco_annotations

    def _create_sub_mask_annotation(self, sub_mask: np.ndarray, image_id: int, category_id: int, annotation_id: int) -> Dict[str, Any]:
        """
        Creates a COCO annotation for a single sub-mask.
        """
        padded_mask = np.pad(sub_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_mask, 0.5)
        
        segmentations = []
        polygons = []

        for contour in contours:
            contour = np.array([(col - 1, row - 1) for row, col in contour])
            poly = Polygon(contour)
            if not poly.is_valid or poly.is_empty:
                continue
            
            if not poly.is_valid or poly.is_empty:
                continue

            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            if len(segmentation) >= 6:
                segmentations.append(segmentation)

        if not polygons:
            return None

        bbox = bbox_from_polygons(polygons)
        multi_poly = MultiPolygon(polygons)
        area = multi_poly.area

        return {
            'segmentation': segmentations,
            'iscrowd': 0,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
        }

    def _save_annotations(self, output_file: str):
        """
        Saves the COCO data to a JSON file.
        """
        save_coco(self.coco_data, output_file)
