from pycocotools import mask as mask_utils
from typing import List
import torch

def determine_polygon_orientation(polygon: List[float]) -> int:
    """
    Determines if a polygon is a positive area (1) or hole (0) based on its orientation.
    
    Args:
        polygon (list): Polygon coordinates in flat format [x1, y1, x2, y2, ...]
        
    Returns:
        int: 1 if a positive area (clockwise), 0 if a hole (counter-clockwise)
    """
    # Convert flat list to coordinate pairs
    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
    if len(points) < 3:
        return 1  # Default to positive area if not enough points
        
    # Calculate signed area using Shoelace formula
    signed_area = sum((points[i][0] * points[(i+1) % len(points)][1]) - 
                      (points[i][1] * points[(i+1) % len(points)][0]) 
                      for i in range(len(points)))
    
    # Clockwise (negative signed area) is positive segment (1)
    # Counter-clockwise (positive signed area) is hole (0)
    return 1 if signed_area < 0 else 0


def create_segmentation_mask(segmentation, segmentation_types, img_height, img_width):
    """
    Create a binary mask from segmentation polygons, properly handling holes and multiple-segments. 
    Note, post_mask/hole_mask and final_mask are inverted shape (H, W).
    
    Args:
        segmentation (list): List of segmentation polygons
        segmentation_types (list): List indicating if each polygon is positive (1) or hole (0)
        img_height (int): Height of the image
        img_width (int): Width of the image
    
    Returns:
        torch.Tensor: Binary mask with holes properly handled
    """
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return None

    # If segmentation_types is None, infer from polygon orientation
    if segmentation_types is None:
        segmentation_types = [determine_polygon_orientation(seg) for seg in segmentation]
        
        # Ensure at least one segment is a positive area (type 1)
        if 1 not in segmentation_types and len(segmentation_types) > 0:
            segmentation_types[0] = 1  # Force first segment to be positive

    final_mask = None
    
    if segmentation_types and len(segmentation_types) == len(segmentation):
        # Handle multi-part segmentation with holes
        positive_segments = [seg for i, seg in enumerate(segmentation) if segmentation_types[i] == 1]
        hole_segments = [seg for i, seg in enumerate(segmentation) if segmentation_types[i] == 0]
        
        if positive_segments:
            # Convert positive segments to RLEs and merge
            pos_rles = mask_utils.frPyObjects(positive_segments, img_height, img_width)
            merged_pos_rle = mask_utils.merge(pos_rles)
            pos_mask = mask_utils.decode(merged_pos_rle)  # shape: (H, W)
            
            if hole_segments:
                # Convert hole segments to RLEs and merge
                hole_rles = mask_utils.frPyObjects(hole_segments, img_height, img_width)
                merged_hole_rle = mask_utils.merge(hole_rles)
                hole_mask = mask_utils.decode(merged_hole_rle)  # shape: (H, W)
                
                # Subtract hole mask from positive mask
                final_mask = pos_mask - hole_mask
            else:
                final_mask = pos_mask
        else:
            return None
    else:
        # Simple segmentation (single part or no type info) - treat as positive
        rles = mask_utils.frPyObjects(segmentation, img_height, img_width)
        final_mask = mask_utils.decode(mask_utils.merge(rles))  # shape: (H, W)
    
    if final_mask is not None:
        return torch.from_numpy(final_mask).float()
    return None
