import json, tempfile, os
import pytest
from cocoutils.utils.categories import CategoryManager

def test_manager_basic():
    cats=[{"id":1,"name":"person"},{"id":2,"name":"car"}]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        json.dump(cats, f)
        f.close()
        mgr = CategoryManager(f.name)
        assert mgr.id_to_name[1] == "person"
        assert mgr.name_to_id["car"] == 2
        assert len(mgr) == 2
        os.unlink(f.name)

def test_manager_file_not_found():
    with pytest.raises(ValueError, match="Categories file not found at:"):
        CategoryManager("non_existent_file.json")

def test_manager_no_filepath():
    with pytest.raises(ValueError, match="A file path to a categories JSON file is required."):
        CategoryManager(None)

def test_duplicate_id():
    cats=[{"id":1,"name":"person"},{"id":1,"name":"car"}]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        json.dump(cats, f)
        f.close()
        with pytest.raises(ValueError, match="Duplicate category ID found: 1"):
            CategoryManager(f.name)
        os.unlink(f.name)

def test_duplicate_name():
    cats=[{"id":1,"name":"person"},{"id":2,"name":"person"}]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        json.dump(cats, f)
        f.close()
        with pytest.raises(ValueError, match="Duplicate category name found: 'person'"):
            CategoryManager(f.name)
        os.unlink(f.name)

def test_missing_key():
    cats=[{"id":1,"name":"person"},{"id":2}]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        json.dump(cats, f)
        f.close()
        with pytest.raises(ValueError, match="Each category must have an 'id' and a 'name'."):
            CategoryManager(f.name)
        os.unlink(f.name)
