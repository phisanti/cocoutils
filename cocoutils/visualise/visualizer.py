import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from pycocotools.coco import COCO
from PIL import Image
from typing import List, Optional, Tuple

class CocoVisualizer:
    """
    Visualizes COCO annotations on images.
    """

    def __init__(self, coco_file: str):
        """
        Initializes the CocoVisualizer.

        Args:
            coco_file (str): Path to the COCO annotations JSON file.
        """
        self.coco = COCO(coco_file)
        self.cat_ids = self.coco.getCatIds()
        self.cats = self.coco.loadCats(self.cat_ids)
        
        # Create a color map for categories
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(self.cats))))
        np.random.shuffle(colors)
        self.catid_to_color = {cat['id']: colors[i % len(colors)] for i, cat in enumerate(self.cats)}

    def visualize(
        self,
        image_path: str,
        image_id: Optional[int] = None,
        ann_ids: Optional[List[int]] = None,
        show_masks: bool = True,
        show_bboxes: bool = True,
        show_class_names: bool = True,
        crop_coords: Optional[Tuple[int, int, int, int]] = None,
        ax: Optional[plt.Axes] = None
    ):
        """
        Visualizes annotations for a given image.

        Args:
            image_path (str): Path to the image file.
            image_id (int, optional): The ID of the image to visualize. If not provided,
                                      it will be inferred from the filename.
            ann_ids (List[int], optional): A list of specific annotation IDs to show. 
                                           If None, all annotations for the image are shown.
            show_masks (bool): Whether to display segmentation masks.
            show_bboxes (bool): Whether to display bounding boxes.
            show_class_names (bool): Whether to display class names.
            crop_coords (Tuple, optional): (x0, y0, x1, y1) coordinates to crop the view.
            ax (plt.Axes, optional): A matplotlib axes object to plot on. If None, a new
                                     figure and axes are created.
        """
        if image_id is None:
            # Infer image_id from filename
            img_ids = self.coco.getImgIds()
            imgs = self.coco.loadImgs(img_ids)
            matching_imgs = [img for img in imgs if img['file_name'] == os.path.basename(image_path)]
            if not matching_imgs:
                raise ValueError(f"Image with filename '{os.path.basename(image_path)}' not found in COCO file.")
            image_id = matching_imgs[0]['id']

        if ann_ids is None:
            ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        
        annotations = self.coco.loadAnns(ann_ids)
        image = np.array(Image.open(image_path))

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(15, 15))
        
        ax.imshow(image)
        ax.axis('off')

        for ann in annotations:
            color = self.catid_to_color.get(ann['category_id'], (1, 0, 0, 1)) # Default to red
            
            if show_bboxes and 'bbox' in ann:
                self._plot_bbox(ax, ann['bbox'], color)
            
            if show_masks and 'segmentation' in ann:
                self._plot_segmentation(ax, ann['segmentation'], color)

            if show_class_names and 'category_id' in ann:
                cat_name = self.coco.loadCats([ann['category_id']])[0]['name']
                x, y, _, _ = ann['bbox']
                ax.text(x, y - 5, cat_name, color='white', fontsize=12, bbox=dict(facecolor=color, alpha=0.7))

        # Create legend
        legend_patches = [mpatches.Patch(color=self.catid_to_color[cat['id']], label=cat['name']) for cat in self.cats]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    def _plot_bbox(self, ax: plt.Axes, bbox: List[float], color):
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    def _plot_segmentation(self, ax: plt.Axes, segmentation: List[List[float]], color):
        for seg in segmentation:
            poly = np.array(seg).reshape((-1, 2))
            ax.add_patch(MplPolygon(poly, closed=True, fill=True, color=color, alpha=0.4))
            ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor=color, linewidth=2))
