"""
COCO file splitter that separates a combined COCO annotation file into 
individual files, one per image.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from ..utils.io import load_coco, save_coco


class CocoSplitter:
    """
    Splits a COCO annotation file into separate files, one per image.
    """

    def __init__(self):
        pass

    def split_file(self, input_path: str, output_dir: str, naming_pattern: str = "{image_name}"):
        """
        Splits a COCO file into separate annotation files, one per image.

        Args:
            input_path (str): Path to the input COCO JSON file.
            output_dir (str): Directory to save the individual COCO files.
            naming_pattern (str): Pattern for output filenames. Available variables:
                - {image_name}: Base name of the image file (without extension)
                - {image_id}: Image ID from COCO file
                Default: "{image_name}"

        Returns:
            List[str]: Paths to the created COCO files.
        """
        coco_data = load_coco(input_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        created_files = []
        
        for image_info in coco_data.get('images', []):
            image_id = image_info['id']
            image_filename = image_info['file_name']
            image_name = Path(image_filename).stem
            
            # Get annotations for this image
            image_annotations = [
                ann for ann in coco_data.get('annotations', [])
                if ann['image_id'] == image_id
            ]
            
            # Create individual COCO structure
            individual_coco = {
                'info': coco_data.get('info', {}),
                'licenses': coco_data.get('licenses', []),
                'images': [image_info],
                'annotations': image_annotations,
                'categories': coco_data.get('categories', [])
            }
            
            # Generate output filename
            output_filename = naming_pattern.format(
                image_name=image_name,
                image_id=image_id
            ) + '.json'
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Save individual COCO file
            save_coco(individual_coco, output_path)
            created_files.append(output_path)
        
        return created_files

    def split_by_categories(self, input_path: str, output_dir: str, 
                           naming_pattern: str = "{image_name}_cat_{category_names}"):
        """
        Splits a COCO file by creating separate files for each unique combination
        of categories present in each image's annotations.

        Args:
            input_path (str): Path to the input COCO JSON file.
            output_dir (str): Directory to save the individual COCO files.
            naming_pattern (str): Pattern for output filenames. Available variables:
                - {image_name}: Base name of the image file (without extension)
                - {image_id}: Image ID from COCO file
                - {category_names}: Underscore-separated category names
                Default: "{image_name}_cat_{category_names}"

        Returns:
            List[str]: Paths to the created COCO files.
        """
        coco_data = load_coco(input_path)
        
        # Create category ID to name mapping
        category_map = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        created_files = []
        
        for image_info in coco_data.get('images', []):
            image_id = image_info['id']
            image_filename = image_info['file_name']
            image_name = Path(image_filename).stem
            
            # Get annotations for this image
            image_annotations = [
                ann for ann in coco_data.get('annotations', [])
                if ann['image_id'] == image_id
            ]
            
            # Get unique categories for this image
            category_ids = set(ann['category_id'] for ann in image_annotations)
            category_names = sorted([category_map.get(cat_id, f"unknown_{cat_id}") 
                                   for cat_id in category_ids])
            category_names_str = "_".join(category_names) if category_names else "no_annotations"
            
            # Create individual COCO structure
            individual_coco = {
                'info': coco_data.get('info', {}),
                'licenses': coco_data.get('licenses', []),
                'images': [image_info],
                'annotations': image_annotations,
                'categories': coco_data.get('categories', [])
            }
            
            # Generate output filename
            output_filename = naming_pattern.format(
                image_name=image_name,
                image_id=image_id,
                category_names=category_names_str
            ) + '.json'
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Save individual COCO file
            save_coco(individual_coco, output_path)
            created_files.append(output_path)
        
        return created_files