"""
COCO health check module for validating COCO format annotation files.

This module provides comprehensive validation and health checking for COCO format
annotation files, including category validation, orphaned annotation detection,
bounding box validation, and statistical summaries.
"""

from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict

from ..utils.io import load_coco


class CocoHealthChecker:
    """
    Health checker for COCO format annotation files.
    
    Provides comprehensive validation including:
    - Category ID validation
    - Orphaned annotation detection
    - Bounding box validation
    - Image-annotation consistency checks
    - Statistical summaries
    """
    
    def __init__(self, coco_path: str):
        """
        Initialize the health checker.
        
        Args:
            coco_path (str): Path to the COCO JSON file.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not valid COCO format.
        """
        self.coco_path = coco_path
        self.coco_data = load_coco(coco_path)
        self.issues: Dict[str, List[str]] = {}
        self.stats: Dict[str, Any] = {}
    
    def run_all_checks(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run all health checks on the COCO file.
        
        Args:
            verbose (bool): If True, include detailed per-image statistics.
            
        Returns:
            Dict[str, Any]: Dictionary containing all check results and statistics.
        """
        results = {
            'basic_stats': self._get_basic_stats(),
            'orphaned_categories': self._check_orphaned_categories(),
            'category_distribution': self._count_annotations_per_class(),
            'multipolygon_stats': self._count_multipolygons(),
            'orphaned_annotations': self._check_orphaned_annotations(),
            'images_without_annotations': self._check_images_without_annotations(),
            'bbox_validation': self._check_bbox_validity(),
        }
        
        if verbose:
            results['per_image_stats'] = self._get_per_image_stats()
        else:
            results['per_image_summary'] = self._get_per_image_summary()
        
        return results
    
    def _get_basic_stats(self) -> Dict[str, int]:
        """Get basic statistics about the COCO dataset."""
        return {
            'n_images': len(self.coco_data.get('images', [])),
            'n_annotations': len(self.coco_data.get('annotations', [])),
            'n_categories': len(self.coco_data.get('categories', []))
        }
    
    def _check_category_ids(self) -> Dict[str, Any]:
        """
        Check that all annotation category_ids exist in the categories map.
        
        Returns:
            Dict with 'valid' (bool) and 'issues' (List[str]).
        """
        valid_category_ids = {cat['id'] for cat in self.coco_data.get('categories', [])}
        
        if not valid_category_ids:
            return {
                'valid': False,
                'issues': ['No categories defined in the dataset'],
                'invalid_count': 0
            }
        
        invalid_annotations = []
        for ann in self.coco_data.get('annotations', []):
            cat_id = ann.get('category_id')
            if cat_id not in valid_category_ids:
                invalid_annotations.append((ann['id'], cat_id))
        
        if invalid_annotations:
            return {
                'valid': False,
                'issues': invalid_annotations,
                'invalid_count': len(invalid_annotations)
            }
        
        return {
            'valid': True,
            'issues': [],
            'invalid_count': 0
        }
    
    def _check_orphaned_categories(self) -> Dict[str, Any]:
        """
        Check for orphaned categories (defined but never used, or used but not defined).
        
        Returns:
            Dict with 'valid' (bool), 'unused_categories' (List), and 'undefined_categories' (Set).
        """
        # Get all defined categories
        defined_categories = {
            cat['id']: cat.get('name', 'unknown')
            for cat in self.coco_data.get('categories', [])
        }
        
        # Get all used categories
        used_category_ids = {
            ann.get('category_id')
            for ann in self.coco_data.get('annotations', [])
        }
        
        # Find unused categories (defined but never used)
        unused_categories = [
            (cat_id, name)
            for cat_id, name in defined_categories.items()
            if cat_id not in used_category_ids
        ]
        
        # Find undefined categories (used but not defined)
        undefined_categories = used_category_ids - set(defined_categories.keys())
        
        return {
            'valid': len(unused_categories) == 0 and len(undefined_categories) == 0,
            'unused_categories': unused_categories,
            'undefined_categories': list(undefined_categories),
            'unused_count': len(unused_categories),
            'undefined_count': len(undefined_categories)
        }
    
    def _count_annotations_per_class(self) -> Dict[int, Dict[str, Any]]:
        """
        Count annotations per category.
        
        Returns:
            Dict mapping category_id to {'name': str, 'count': int}.
        """
        category_info = {
            cat['id']: {'name': cat.get('name', 'unknown'), 'count': 0}
            for cat in self.coco_data.get('categories', [])
        }
        
        for ann in self.coco_data.get('annotations', []):
            cat_id = ann.get('category_id')
            if cat_id in category_info:
                category_info[cat_id]['count'] += 1
        
        return category_info
    
    def _count_multipolygons(self) -> Dict[str, Any]:
        """
        Count annotations with multiple polygons in segmentation.
        
        Returns:
            Dict with 'count' and 'details' (list of tuples).
        """
        multipolygon_count = 0
        multipolygon_details = []
        
        for ann in self.coco_data.get('annotations', []):
            segmentation = ann.get('segmentation', [])
            if isinstance(segmentation, list) and len(segmentation) > 1:
                multipolygon_count += 1
                multipolygon_details.append((
                    ann['id'],
                    ann.get('image_id'),
                    len(segmentation)
                ))
        
        return {
            'count': multipolygon_count,
            'details': multipolygon_details
        }
    
    def _check_orphaned_annotations(self) -> Dict[str, Any]:
        """
        Check for annotations referencing non-existent images.
        
        Returns:
            Dict with 'valid' (bool) and 'orphaned_ids' (List[int]).
        """
        valid_image_ids = {img['id'] for img in self.coco_data.get('images', [])}
        
        orphaned = [
            ann['id'] for ann in self.coco_data.get('annotations', [])
            if ann.get('image_id') not in valid_image_ids
        ]
        
        return {
            'valid': len(orphaned) == 0,
            'orphaned_ids': orphaned,
            'count': len(orphaned)
        }
    
    def _check_images_without_annotations(self) -> Dict[str, Any]:
        """
        Find images with no annotations.
        
        Returns:
            Dict with 'count' and 'images' (list of tuples).
        """
        images_with_anns = {ann['image_id'] for ann in self.coco_data.get('annotations', [])}
        
        images_without_anns = [
            (img['id'], img.get('file_name', 'unknown'))
            for img in self.coco_data.get('images', [])
            if img['id'] not in images_with_anns
        ]
        
        return {
            'count': len(images_without_anns),
            'images': images_without_anns
        }
    
    def _check_bbox_validity(self) -> Dict[str, Any]:
        """
        Check for invalid bounding boxes.
        
        Returns:
            Dict with 'valid' (bool) and 'invalid_bboxes' (list of tuples).
        """
        invalid_bboxes = []
        
        for ann in self.coco_data.get('annotations', []):
            bbox = ann.get('bbox')
            if not bbox or len(bbox) != 4:
                invalid_bboxes.append((ann['id'], 'missing or malformed'))
                continue
            
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                invalid_bboxes.append((ann['id'], f'invalid dimensions: w={w}, h={h}'))
            elif x < 0 or y < 0:
                invalid_bboxes.append((ann['id'], f'negative coordinates: x={x}, y={y}'))
        
        return {
            'valid': len(invalid_bboxes) == 0,
            'invalid_bboxes': invalid_bboxes,
            'count': len(invalid_bboxes)
        }
    
    def _get_per_image_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get detailed statistics per image.
        
        Returns:
            Dict mapping image_id to stats dict.
        """
        image_info = {
            img['id']: {
                'file_name': img.get('file_name', 'unknown'),
                'ann_count': 0,
                'multipolygon_count': 0
            }
            for img in self.coco_data.get('images', [])
        }
        
        for ann in self.coco_data.get('annotations', []):
            img_id = ann.get('image_id')
            if img_id in image_info:
                image_info[img_id]['ann_count'] += 1
                
                segmentation = ann.get('segmentation', [])
                if isinstance(segmentation, list) and len(segmentation) > 1:
                    image_info[img_id]['multipolygon_count'] += 1
        
        return image_info
    
    def _get_per_image_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all images.
        
        Returns:
            Dict with min, max, avg for annotations and multipolygons per image.
        """
        image_info = self._get_per_image_stats()
        
        if not image_info:
            return {}
        
        ann_counts = [info['ann_count'] for info in image_info.values()]
        mp_counts = [info['multipolygon_count'] for info in image_info.values()]
        
        summary = {
            'annotations': {
                'min': min(ann_counts) if ann_counts else 0,
                'max': max(ann_counts) if ann_counts else 0,
                'avg': sum(ann_counts) / len(ann_counts) if ann_counts else 0
            }
        }
        
        if any(mp_counts):
            summary['multipolygons'] = {
                'min': min(mp_counts),
                'max': max(mp_counts),
                'avg': sum(mp_counts) / len(mp_counts)
            }
        
        return summary
