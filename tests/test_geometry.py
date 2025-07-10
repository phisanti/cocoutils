from cocoutils.utils.geometry import determine_polygon_orientation, create_segmentation_mask
import numpy as np
import torch

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
