# cocoutils

A toolkit for working with COCO annotations.

`cocoutils` is a Python library and command-line tool for converting segmentation masks to COCO format, reconstructing masks from COCO annotations, merging multiple COCO annotation files, splitting COCO files into individual per-image files, and visualizing annotations.

## Overview

This project provides a set of tools for working with COCO annotations. It is designed to be a flexible and easy-to-use solution for common COCO-related tasks. The project is structured as a Python package with a modular design, allowing you to use the components you need.

## Current Capabilities

- **Convert**: Convert segmentation masks (in TIFF format) to COCO JSON format.
- **Reconstruct**: Reconstruct segmentation masks from a COCO JSON file.
- **Merge**: Merge two COCO annotation files into a single file.
- **Split**: Split a combined COCO file into individual files, one per image.
- **Visualise**: Visualize COCO annotations on an image.

## Installation

To install the package, clone the repository and install it in editable mode using pip:

```bash
git clone https://github.com/phisanti/cocoutils
cd cocoutils
pip install -e .
```

# cocoutils

A toolkit for working with COCO annotations.

`cocoutils` is a Python library **and** command-line tool for

* converting segmentation masks **(TIFF)** to COCO JSON  
* reconstructing masks from COCO JSON  
* merging multiple COCO files  
* splitting combined COCO files into individual per-image files
* visualising annotations (regular view or background-masked view)

---

## Installation

```bash
git clone <repository-url>
cd cocoutils
pip install -e .
```

---

## Command-line interface

All commands are exposed through the single entry-point `cocoutils`.

### `convert`

Convert a directory of labelled TIFF masks to a COCO file.

```bash
cocoutils convert \
  --input-dir   /path/to/masks \
  --output-file /path/to/coco.json \
  --categories  /path/to/categories.json
```

### `reconstruct`

Recreate individual mask images from a COCO file.

```bash
cocoutils reconstruct \
  --input-file  /path/to/coco.json \
  --output-dir  /path/to/reconstructed-masks \
  --workers     4        # 0 = use all CPU cores, 1 = sequential
```

### `merge`

Combine two compatible COCO files, automatically remapping IDs to avoid
collisions.

```bash
cocoutils merge \
  --file1       /path/to/coco1.json \
  --file2       /path/to/coco2.json \
  --output-file /path/to/merged.json
```

### `split`

Split a combined COCO file into individual files, one per image.

```bash
# Basic split - one file per image
cocoutils split \
  --input-file  /path/to/combined.json \
  --output-dir  /path/to/split-files

# Custom naming pattern
cocoutils split \
  --input-file      /path/to/combined.json \
  --output-dir      /path/to/split-files \
  --naming-pattern  "img_{image_id}_{image_name}"

# Include category information in filenames
cocoutils split \
  --input-file   /path/to/combined.json \
  --output-dir   /path/to/split-files \
  --by-categories
```

### `visualise`

Draw annotations on top of an image (with optional masked view).

```bash
# Standard view
cocoutils visualise \
  --coco-file  /path/to/coco.json \
  --image-path /path/to/image.png

# Masked view – background set to 0 for the selected annotations
cocoutils visualise \
  --coco-file       /path/to/coco.json \
  --image-path      /path/to/image.png \
  --masked-view \
  --annotation-ids "1,2,3"
```

---

## Using as a Python library

```python
from cocoutils.convert   import CocoConverter
from cocoutils.reconstruct import CocoReconstructor
from cocoutils.merge     import CocoMerger
from cocoutils.split     import CocoSplitter
from cocoutils.visualise import CocoVisualizer

# Convert
converter = CocoConverter(categories_path="categories.json")
converter.from_masks("masks_dir", "annotations.json")

# Reconstruct
reconstructor = CocoReconstructor()
reconstructor.from_coco("annotations.json", "reconstructed_masks", workers=4)

# Merge
merger = CocoMerger()
merger.merge_files("a.json", "b.json", "merged.json")

# Split
splitter = CocoSplitter()
splitter.split_file("combined.json", "split_output")

# Visualise
viz = CocoVisualizer("annotations.json")
viz.visualize("image.png")
```

---

## Category definitions

`convert` requires a **categories JSON** file whose content looks like:

```json
[
  {"id": 1, "name": "cell"},
  {"id": 2, "name": "nucleus"}
]
```

IDs must be consecutive positive integers and names unique.

---
## License

EULA