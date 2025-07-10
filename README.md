# cocoutils

A toolkit for working with COCO annotations.

`cocoutils` is a Python library and command-line tool for converting segmentation masks to COCO format, reconstructing masks from COCO annotations, merging multiple COCO annotation files, and visualizing annotations.

## Overview

This project provides a set of tools for working with COCO annotations. It is designed to be a flexible and easy-to-use solution for common COCO-related tasks. The project is structured as a Python package with a modular design, allowing you to use the components you need.

## Current Capabilities

- **Convert**: Convert segmentation masks (in TIFF format) to COCO JSON format.
- **Reconstruct**: Reconstruct segmentation masks from a COCO JSON file.
- **Merge**: Merge two COCO annotation files into a single file.
- **Visualise**: Visualize COCO annotations on an image.

## Installation

To install the package, clone the repository and install it in editable mode using pip:

```bash
git clone <repository-url>
cd cocoutils
pip install -e .
```

## Usage

`cocoutils` can be used as a command-line tool or as a Python library.

### Command-Line Interface

The command-line interface is built with Typer and provides the following commands:

#### `convert`

Converts segmentation masks to COCO format.

```bash
cocoutils convert --input-dir /path/to/masks --output-file /path/to/coco.json --categories /path/to/categories.json
```

#### `reconstruct`

Reconstructs segmentation masks from a COCO JSON file.

```bash
cocoutils reconstruct --coco-file /path/to/coco.json --output-dir /path/to/reconstructed-masks
```

#### `merge`

Merges two COCO annotation files into a single file.

```bash
cocoutils merge --file1 /path/to/coco1.json --file2 /path/to/coco2.json --output-file /path/to/merged.json
```

#### `visualise`

Visualizes COCO annotations on an image.

```bash
cocoutils visualise --coco-file /path/to/coco.json --image-path /path/to/image.png
```