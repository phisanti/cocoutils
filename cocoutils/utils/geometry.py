from pycocotools import mask as mask_utils
from typing import List
import torch
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage import measure

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


def bbox_from_polygons(polygons: List[Polygon]) -> List[float]:
    """
    Compute COCO-style bbox [x, y, w, h] from a list of Shapely polygons.
    
    Args:
        polygons (List[Polygon]): List of Shapely Polygon objects.
        
    Returns:
        List[float]: Bounding box in COCO format [x, y, width, height].
                    Returns [0, 0, 0, 0] if list is empty or all polygons are invalid.
    """
    if not polygons:
        return [0.0, 0.0, 0.0, 0.0]
    
    # Filter out invalid polygons
    valid_polygons = [p for p in polygons if p.is_valid and not p.is_empty]
    
    if not valid_polygons:
        return [0.0, 0.0, 0.0, 0.0]
    
    # Create union of all polygons to get overall bounds
    if len(valid_polygons) == 1:
        combined = valid_polygons[0]
    else:
        combined = unary_union(valid_polygons)
    
    # Get bounds: (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = combined.bounds
    
    # Convert to COCO format [x, y, width, height]
    width = max(0.0, maxx - minx)
    height = max(0.0, maxy - miny)
    
    return [float(minx), float(miny), float(width), float(height)]


def extract_polygon_segments(mask) -> List[List[float]]:
    """
    Extract polygon segments from a binary mask with proper clockwise orientation.
    
    This is the core polygon extraction logic that should be used consistently
    across all COCO annotation creation methods.
    
    Args:
        mask: Binary mask (2D array) representing a single object or component
        
    Returns:
        List of polygon segments as flat coordinate lists [x1, y1, x2, y2, ...]
        Each segment is guaranteed to have clockwise orientation for positive areas.
    """
    if np.sum(mask) < 10:  # Skip tiny objects
        return []
    
    # Add padding around mask to handle border objects
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    
    # Find contours using scikit-image
    contours = measure.find_contours(padded_mask, 0.5)
    
    segments = []
    
    for contour in contours:
        # Convert from (row, col) to (x, y) and subtract padding
        points = [(col - 1, row - 1) for row, col in contour]
        
        if len(points) < 3:
            continue
            
        try:
            poly = Polygon(points)
            if not poly.is_valid or poly.is_empty:
                continue
                
            # Simplify polygon but preserve topology
            poly = poly.simplify(1.0, preserve_topology=True)
            if poly.is_empty or not poly.is_valid:
                continue
            
            # Extract coordinates
            coords = np.array(poly.exterior.coords).ravel().tolist()
            if len(coords) >= 6:  # At least 3 points
                segments.append(coords)
                
        except Exception:
            continue
    
    return segments


def extract_bbox_from_segments(segments: List[List[float]]) -> List[float]:
    """
    Extract bounding box from polygon segments.
    
    Optimized implementation using numpy operations for better performance.
    
    Args:
        segments: List of polygon segments as flat coordinate lists
        
    Returns:
        Bounding box as [x, y, width, height]
    """
    if not segments:
        return [0.0, 0.0, 0.0, 0.0]
    
    # Collect all coordinates using numpy for efficient processing
    all_coords = []
    for seg in segments:
        if len(seg) >= 6:  # At least 3 points
            try:
                coords = np.array(seg).reshape(-1, 2)
                all_coords.append(coords)
            except (ValueError, TypeError):
                continue
    
    if not all_coords:
        return [0.0, 0.0, 0.0, 0.0]
    
    # Combine all coordinates and find bounds
    try:
        combined = np.vstack(all_coords)
        minx, miny = np.min(combined, axis=0)
        maxx, maxy = np.max(combined, axis=0)
        
        width = max(0.0, float(maxx - minx))
        height = max(0.0, float(maxy - miny))
        
        return [float(minx), float(miny), float(width), float(height)]
    except (ValueError, TypeError):
        return [0.0, 0.0, 0.0, 0.0]


def extract_area_from_segments(segments: List[List[float]]) -> float:
    """
    Extract total area from polygon segments.
    
    Optimized implementation using numpy-based shoelace formula for better performance.
    
    Args:
        segments: List of polygon segments as flat coordinate lists
        
    Returns:
        Total area of all segments combined
    """
    if not segments:
        return 0.0
    
    total_area = 0.0
    for seg in segments:
        if len(seg) >= 6:  # At least 3 points
            try:
                coords = np.array(seg).reshape(-1, 2)
                x = coords[:, 0]
                y = coords[:, 1]
                
                # Shoelace formula for polygon area
                # Area = 0.5 * |sum(x[i] * y[i+1] - x[i+1] * y[i])|
                area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + 
                                (x[-1] * y[0] - x[0] * y[-1]))
                total_area += area
            except (ValueError, TypeError, IndexError):
                continue
    
    return float(total_area)
