import typer
from typing_extensions import Annotated
import os

from .convert import CocoConverter
from .reconstruct import CocoReconstructor
from .merge import CocoMerger
from .visualise import CocoVisualizer
import matplotlib.pyplot as plt

app = typer.Typer(help="A toolkit for COCO annotation generation, conversion, merging, and visualisation.")

@app.command()
def convert(
    input_dir: Annotated[str, typer.Option("--input-dir", "-i", help="Path to directory containing classified object TIFF images")],
    output_file: Annotated[str, typer.Option("--output-file", "-o", help="Path to save the COCO annotations JSON file")],
    categories: Annotated[str, typer.Option("--categories", "-c", help="Path to the categories JSON file")]
):
    """
    Converts segmentation masks to COCO format.
    """
    print(f"Converting masks from '{input_dir}' to COCO format at '{output_file}'...")
    converter = CocoConverter(categories_path=categories)
    converter.from_masks(input_dir=input_dir, output_file=output_file)
    print("Conversion complete.")

@app.command()
def reconstruct(
    coco_file: Annotated[str, typer.Option("--input-file", "-i", help="Path to COCO annotations JSON file")],
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Directory to save the generated mask images")],
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of parallel workers")] = 0
):
    """
    Reconstructs mask images from a COCO annotation file.
    """
    print(f"Reconstructing masks from '{coco_file}' to '{output_dir}'...")
    reconstructor = CocoReconstructor()
    reconstructor.from_coco(coco_file, output_dir, workers)
    print("Reconstruction complete.")

@app.command()
def merge(
    file1: Annotated[str, typer.Option(help="Path to the first COCO file")],
    file2: Annotated[str, typer.Option(help="Path to the second COCO file")],
    output_file: Annotated[str, typer.Option(help="Path to save the combined COCO file")]
):
    """
    Merges two COCO annotation files.
    """
    print(f"Merging '{file1}' and '{file2}' into '{output_file}'...")
    try:
        merger = CocoMerger()
        merger.merge_files(file1, file2, output_file)
        print("Merge complete.")
    except ValueError as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)

@app.command()
def visualise(
    coco_file: Annotated[str, typer.Option("--coco-file", "-c", help="Path to the COCO annotations JSON file")],
    image_path: Annotated[str, typer.Option("--image-path", "-i", help="Path to the image file to visualize")],
    no_masks: Annotated[bool, typer.Option("--no-masks", "-m", help="Do not display segmentation masks")] = False,
    no_bboxes: Annotated[bool, typer.Option("--no-bboxes", "-b", help="Do not display bounding boxes")] = False,
    no_class_names: Annotated[bool, typer.Option("--no-class-names", "-n", help="Do not display class names")] = False,
    masked_view: Annotated[bool, typer.Option("--masked-view", help="Show masked visualization (background pixels set to 0)")] = False,
    annotation_ids: Annotated[str, typer.Option("--annotation-ids", help="Comma-separated annotation IDs for masked view (e.g., '1,2,3')")] = None
):
    """
    Visualizes COCO annotations on an image.
    """
    print(f"Visualizing annotations from '{coco_file}' on '{image_path}'...")
    try:
        visualizer = CocoVisualizer(coco_file=coco_file)
        fig, ax = plt.subplots(1, figsize=(15, 15))
        
        if masked_view:
            # Handle masked visualization
            if annotation_ids is None:
                print("Error: --annotation-ids is required when using --masked-view")
                raise typer.Exit(code=1)
            
            # Parse annotation IDs
            try:
                ids = [int(id.strip()) for id in annotation_ids.split(',')]
            except ValueError:
                print("Error: Invalid annotation IDs format. Use comma-separated integers (e.g., '1,2,3')")
                raise typer.Exit(code=1)
            
            # Load image for masked visualization
            from PIL import Image
            import numpy as np
            image = np.array(Image.open(image_path))
            
            visualizer.visualize_annotations_masked(
                image=image,
                annotation_ids=ids,
                ax=ax,
                show=False
            )
        else:
            # Standard visualization
            visualizer.visualize(
                image_path=image_path,
                show_masks=not no_masks,
                show_bboxes=not no_bboxes,
                show_class_names=not no_class_names,
                ax=ax
            )
        
        plt.tight_layout()
        plt.show()
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
