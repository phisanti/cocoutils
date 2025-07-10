import importlib

def test_import_packages():
    for mod in ["cocoutils", "cocoutils.utils", "cocoutils.convert", "cocoutils.reconstruct", "cocoutils.merge", "cocoutils.visualise"]:
        assert importlib.import_module(mod)


def test_import_bbox_from_polygons():
    """Test that bbox_from_polygons can be imported from utils package."""
    from cocoutils.utils import bbox_from_polygons
    assert callable(bbox_from_polygons)
