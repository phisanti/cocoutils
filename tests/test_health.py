"""Tests for the health check module."""

import json
import pytest
import tempfile
from pathlib import Path

from cocoutils.health import CocoHealthChecker
from cocoutils.health.formatters import HumanFormatter, TokenOptimizedFormatter


@pytest.fixture
def valid_coco_data():
    """Fixture for valid COCO data."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640},
            {"id": 2, "file_name": "image2.jpg", "height": 480, "width": 640}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 100, 100],
                "area": 10000,
                "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [200, 200, 50, 50],
                "area": 2500,
                "segmentation": [[200, 200, 250, 200, 250, 250, 200, 250]],
                "iscrowd": 0
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50, 50, 80, 80],
                "area": 6400,
                "segmentation": [[50, 50, 130, 50, 130, 130, 50, 130]],
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"},
            {"id": 2, "name": "dog", "supercategory": "animal"}
        ]
    }


@pytest.fixture
def coco_with_multipolygon():
    """Fixture for COCO data with multipolygon annotations."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 200, 200],
                "area": 30000,
                "segmentation": [
                    [10, 10, 110, 10, 110, 110, 10, 110],
                    [150, 150, 210, 150, 210, 210, 150, 210]
                ],
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "object", "supercategory": "thing"}
        ]
    }


@pytest.fixture
def coco_with_invalid_category_ids():
    """Fixture for COCO data with invalid category IDs."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 99,  # Invalid category ID
                "bbox": [10, 10, 100, 100],
                "area": 10000,
                "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"}
        ]
    }


@pytest.fixture
def coco_with_orphaned_annotations():
    """Fixture for COCO data with orphaned annotations."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 99,  # Invalid image ID
                "category_id": 1,
                "bbox": [10, 10, 100, 100],
                "area": 10000,
                "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"}
        ]
    }


@pytest.fixture
def coco_with_invalid_bboxes():
    """Fixture for COCO data with invalid bounding boxes."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, -50, 100],  # Negative width
                "area": 10000,
                "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [-10, 10, 100, 100],  # Negative x coordinate
                "area": 10000,
                "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"}
        ]
    }


@pytest.fixture
def coco_with_image_no_annotations():
    """Fixture for COCO data with images without annotations."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640},
            {"id": 2, "file_name": "image2.jpg", "height": 480, "width": 640}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 100, 100],
                "area": 10000,
                "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"}
        ]
    }


@pytest.fixture
def coco_with_unused_categories():
    """Fixture for COCO data with unused categories."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 100, 100],
                "area": 10000,
                "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"},
            {"id": 2, "name": "dog", "supercategory": "animal"},  # Unused
            {"id": 3, "name": "bird", "supercategory": "animal"}  # Unused
        ]
    }


@pytest.fixture
def coco_with_undefined_categories():
    """Fixture for COCO data with undefined categories in use."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 100, 100],
                "area": 10000,
                "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 99,  # Undefined
                "bbox": [200, 200, 50, 50],
                "area": 2500,
                "segmentation": [[200, 200, 250, 200, 250, 250, 200, 250]],
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"}
        ]
    }


def create_temp_coco_file(data):
    """Helper to create temporary COCO file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, temp_file)
    temp_file.close()
    return temp_file.name


class TestCocoHealthChecker:
    """Tests for CocoHealthChecker class."""
    
    def test_valid_coco_file(self, valid_coco_data):
        """Test health check on valid COCO file."""
        temp_path = create_temp_coco_file(valid_coco_data)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            assert results['basic_stats']['n_images'] == 2
            assert results['basic_stats']['n_annotations'] == 3
            assert results['basic_stats']['n_categories'] == 2
            assert results['orphaned_categories']['valid'] is True
            assert results['orphaned_annotations']['valid'] is True
            assert results['bbox_validation']['valid'] is True
        finally:
            Path(temp_path).unlink()
    
    def test_multipolygon_detection(self, coco_with_multipolygon):
        """Test detection of multipolygon annotations."""
        temp_path = create_temp_coco_file(coco_with_multipolygon)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            assert results['multipolygon_stats']['count'] == 1
            assert len(results['multipolygon_stats']['details']) == 1
            ann_id, img_id, poly_count = results['multipolygon_stats']['details'][0]
            assert ann_id == 1
            assert img_id == 1
            assert poly_count == 2
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_category_ids(self, coco_with_invalid_category_ids):
        """Test detection of invalid category IDs (now part of orphaned categories check)."""
        temp_path = create_temp_coco_file(coco_with_invalid_category_ids)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            # Invalid category IDs are now detected as undefined categories
            assert results['orphaned_categories']['valid'] is False
            assert results['orphaned_categories']['undefined_count'] == 1
            assert 99 in results['orphaned_categories']['undefined_categories']
        finally:
            Path(temp_path).unlink()
    
    def test_orphaned_annotations(self, coco_with_orphaned_annotations):
        """Test detection of orphaned annotations."""
        temp_path = create_temp_coco_file(coco_with_orphaned_annotations)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            assert results['orphaned_annotations']['valid'] is False
            assert results['orphaned_annotations']['count'] == 1
            assert 1 in results['orphaned_annotations']['orphaned_ids']
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_bboxes(self, coco_with_invalid_bboxes):
        """Test detection of invalid bounding boxes."""
        temp_path = create_temp_coco_file(coco_with_invalid_bboxes)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            assert results['bbox_validation']['valid'] is False
            assert results['bbox_validation']['count'] == 2
        finally:
            Path(temp_path).unlink()
    
    def test_images_without_annotations(self, coco_with_image_no_annotations):
        """Test detection of images without annotations."""
        temp_path = create_temp_coco_file(coco_with_image_no_annotations)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            assert results['images_without_annotations']['count'] == 1
            img_id, filename = results['images_without_annotations']['images'][0]
            assert img_id == 2
            assert filename == "image2.jpg"
        finally:
            Path(temp_path).unlink()
    
    def test_category_distribution(self, valid_coco_data):
        """Test counting of annotations per category."""
        temp_path = create_temp_coco_file(valid_coco_data)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            cat_dist = results['category_distribution']
            assert cat_dist[1]['name'] == 'cat'
            assert cat_dist[1]['count'] == 2
            assert cat_dist[2]['name'] == 'dog'
            assert cat_dist[2]['count'] == 1
        finally:
            Path(temp_path).unlink()
    
    def test_per_image_stats_verbose(self, valid_coco_data):
        """Test per-image statistics in verbose mode."""
        temp_path = create_temp_coco_file(valid_coco_data)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks(verbose=True)
            
            assert 'per_image_stats' in results
            image_stats = results['per_image_stats']
            assert image_stats[1]['ann_count'] == 2
            assert image_stats[2]['ann_count'] == 1
        finally:
            Path(temp_path).unlink()
    
    def test_per_image_summary_non_verbose(self, valid_coco_data):
        """Test per-image summary in non-verbose mode."""
        temp_path = create_temp_coco_file(valid_coco_data)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks(verbose=False)
            
            assert 'per_image_summary' in results
            summary = results['per_image_summary']
            assert summary['annotations']['min'] == 1
            assert summary['annotations']['max'] == 2
        finally:
            Path(temp_path).unlink()
    
    def test_unused_categories(self, coco_with_unused_categories):
        """Test detection of unused categories."""
        temp_path = create_temp_coco_file(coco_with_unused_categories)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            assert results['orphaned_categories']['valid'] is False
            assert results['orphaned_categories']['unused_count'] == 2
            unused_ids = [cat_id for cat_id, name in results['orphaned_categories']['unused_categories']]
            assert 2 in unused_ids
            assert 3 in unused_ids
        finally:
            Path(temp_path).unlink()
    
    def test_undefined_categories(self, coco_with_undefined_categories):
        """Test detection of undefined categories being used."""
        temp_path = create_temp_coco_file(coco_with_undefined_categories)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            assert results['orphaned_categories']['valid'] is False
            assert results['orphaned_categories']['undefined_count'] == 1
            assert 99 in results['orphaned_categories']['undefined_categories']
        finally:
            Path(temp_path).unlink()
    
    def test_no_orphaned_categories(self, valid_coco_data):
        """Test that valid data has no orphaned categories."""
        temp_path = create_temp_coco_file(valid_coco_data)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            
            assert results['orphaned_categories']['valid'] is True
            assert results['orphaned_categories']['unused_count'] == 0
            assert results['orphaned_categories']['undefined_count'] == 0
        finally:
            Path(temp_path).unlink()


class TestFormatters:
    """Tests for formatter classes."""
    
    def test_human_formatter(self, valid_coco_data):
        """Test human-readable formatter."""
        temp_path = create_temp_coco_file(valid_coco_data)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            formatter = HumanFormatter(results, temp_path)
            output = formatter.format()
            
            assert "COCO Health Check" in output
            assert "Basic Statistics" in output
            assert "Orphaned Category Check" in output
            assert "OK" in output or "ERROR" in output
            assert "=" in output  # Section separators
        finally:
            Path(temp_path).unlink()
    
    def test_token_optimized_formatter(self, valid_coco_data):
        """Test token-optimized formatter."""
        temp_path = create_temp_coco_file(valid_coco_data)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            formatter = TokenOptimizedFormatter(results, temp_path)
            output = formatter.format()
            
            assert "COCO_HEALTH_CHECK" in output
            assert "STATS:" in output
            assert "ORPHANED_CATEGORIES:" in output
            assert "CLASS_DISTRIBUTION:" in output
            # Should be more compact than human format
            assert len(output) < len(HumanFormatter(results, temp_path).format())
        finally:
            Path(temp_path).unlink()
    
    def test_human_formatter_with_errors(self, coco_with_invalid_category_ids):
        """Test human formatter with validation errors."""
        temp_path = create_temp_coco_file(coco_with_invalid_category_ids)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            formatter = HumanFormatter(results, temp_path)
            output = formatter.format()
            
            assert "ERROR" in output
            assert "undefined categories" in output or "Category ID" in output
        finally:
            Path(temp_path).unlink()
    
    def test_token_formatter_with_errors(self, coco_with_orphaned_annotations):
        """Test token-optimized formatter with errors."""
        temp_path = create_temp_coco_file(coco_with_orphaned_annotations)
        try:
            checker = CocoHealthChecker(temp_path)
            results = checker.run_all_checks()
            formatter = TokenOptimizedFormatter(results, temp_path)
            output = formatter.format()
            
            assert "ERROR" in output
            assert "ORPHANED_ANNOTATIONS: ERROR" in output
            assert "count=1" in output
        finally:
            Path(temp_path).unlink()
