import json
from typing import List, Dict, Any

class CategoryManager:
    """
    Manages COCO categories from a JSON file.

    Args:
        filepath (str): Path to the categories JSON file. The file should contain a list of dictionaries,
                        each with "id" and "name" keys.

    Raises:
        ValueError: If filepath is None, the file is not found, or if there are duplicate IDs or names.
    """
    def __init__(self, filepath: str):
        if not filepath:
            raise ValueError("A file path to a categories JSON file is required.")
        
        try:
            with open(filepath, 'r') as f:
                self.categories: List[Dict[str, Any]] = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Categories file not found at: {filepath}")

        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        
        ids = set()
        names = set()

        for category in self.categories:
            cat_id = category.get("id")
            cat_name = category.get("name")

            if cat_id is None or cat_name is None:
                raise ValueError("Each category must have an 'id' and a 'name'.")

            if cat_id in ids:
                raise ValueError(f"Duplicate category ID found: {cat_id}")
            if cat_name in names:
                raise ValueError(f"Duplicate category name found: '{cat_name}'")
            
            ids.add(cat_id)
            names.add(cat_name)
            
            self.id_to_name[cat_id] = cat_name
            self.name_to_id[cat_name] = cat_id

    def __len__(self) -> int:
        return len(self.categories)

    def get_category_id(self, name: str) -> int:
        """Get category ID by name."""
        if name not in self.name_to_id:
            raise ValueError(f"Category '{name}' not found.")
        return self.name_to_id[name]

    def get_category_name(self, cat_id: int) -> str:
        """Get category name by ID."""
        if cat_id not in self.id_to_name:
            raise ValueError(f"Category ID {cat_id} not found.")
        return self.id_to_name[cat_id]
