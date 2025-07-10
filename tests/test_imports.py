import importlib

def test_import_packages():
    for mod in ["cocoutils", "cocoutils.utils", "cocoutils.convert", "cocoutils.reconstruct", "cocoutils.merge", "cocoutils.visualise"]:
        assert importlib.import_module(mod)
