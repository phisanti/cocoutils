from cocoutils.utils.geometry import determine_polygon_orientation, create_segmentation_mask, bbox_from_polygons
import numpy as np
import torch
from shapely.geometry import Polygon

def test_orientation_clockwise():
    poly = [0, 0, 0, 1, 1, 1, 1, 0]  # square clockwise
    assert determine_polygon_orientation(poly) == 1

def test_orientation_counter_clockwise():
    poly=[0,0, 1,0, 1,1, 0,1]  # square CCW gives positive area -> returns 0
    assert determine_polygon_orientation(poly)==0

def test_create_segmentation_mask_simple():
    segmentation = [[0, 0, 0, 10, 10, 10, 10, 0]]
    segmentation_types = [1]
    mask = create_segmentation_mask(segmentation, segmentation_types, 20, 20)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (20, 20)
    assert mask.sum() > 0
    assert mask[5,5] == 1
    assert mask[15,15] == 0

def test_create_segmentation_mask_with_hole():
    segmentation = [
        [0, 0, 0, 20, 20, 20, 20, 0],  # Outer polygon
        [5, 5, 5, 15, 15, 15, 15, 5]   # Inner hole
    ]
    segmentation_types = [1, 0]
    mask = create_segmentation_mask(segmentation, segmentation_types, 25, 25)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (25, 25)
    assert mask.sum() > 0
    assert mask[10, 10] == 0  # Inside the hole
    assert mask[2, 2] == 1    # Outside the hole, inside the polygon
    assert mask[22, 22] == 0  # Outside the polygon


def test_bbox_from_polygons_empty():
    """Test bbox_from_polygons with empty list returns zeros."""
    result = bbox_from_polygons([])
    assert result == [0.0, 0.0, 0.0, 0.0]


def test_bbox_from_polygons_single_square():
    """Test bbox_from_polygons with single square polygon."""
    # Create a 10x10 square starting at (5, 5)
    square = Polygon([(5, 5), (5, 15), (15, 15), (15, 5)])
    result = bbox_from_polygons([square])
    
    # Expected bbox: [x=5, y=5, width=10, height=10]
    expected = [5.0, 5.0, 10.0, 10.0]
    assert result == expected


def test_bbox_from_polygons_multiple_squares():
    """Test bbox_from_polygons with two separate squares."""
    # Create two squares: one at (0,0) to (10,10), another at (15,15) to (25,25)
    square1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    square2 = Polygon([(15, 15), (15, 25), (25, 25), (25, 15)])
    result = bbox_from_polygons([square1, square2])
    
    # Expected bbox should encompass both squares: [x=0, y=0, width=25, height=25]
    expected = [0.0, 0.0, 25.0, 25.0]
    assert result == expected


def test_bbox_from_polygons_with_invalid():
    """Test bbox_from_polygons filters out invalid polygons."""
    # Create one valid square and one invalid (empty) polygon
    valid_square = Polygon([(0, 0), (0, 5), (5, 5), (5, 0)])
    invalid_poly = Polygon()  # Empty polygon
    
    result = bbox_from_polygons([valid_square, invalid_poly])
    
    # Should only consider the valid square
    expected = [0.0, 0.0, 5.0, 5.0]
    assert result == expected


def test_bbox_from_polygons_all_invalid():
    """Test bbox_from_polygons with all invalid polygons returns zeros."""
    invalid_poly1 = Polygon()  # Empty polygon
    # Create a self-intersecting polygon (invalid)
    invalid_poly2 = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])  # Self-intersecting
    
    result = bbox_from_polygons([invalid_poly1, invalid_poly2])
    assert result == [0.0, 0.0, 0.0, 0.0]
