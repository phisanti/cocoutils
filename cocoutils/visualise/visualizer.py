import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from pycocotools.coco import COCO
from PIL import Image
import torch
from typing import List, Optional, Tuple, Union
from ..utils.geometry import create_segmentation_mask, determine_polygon_orientation

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
        """
        Plot segmentation polygons with proper hole handling using matplotlib Path objects.
        Creates a single filled polygon with holes properly cut out.
        """
        if not segmentation:
            return
            
        # Separate positive areas from holes based on orientation
        positive_polygons = []
        hole_polygons = []
        
        for seg in segmentation:
            orientation = determine_polygon_orientation(seg)
            poly_coords = np.array(seg).reshape((-1, 2))
            
            if orientation == 1:  # Positive area (clockwise)
                positive_polygons.append(poly_coords)
            else:  # Hole (counter-clockwise)
                hole_polygons.append(poly_coords)
        
        # Create a single path with all positive polygons and holes
        if positive_polygons:
            if hole_polygons:
                # Create a path with holes
                vertices = []
                codes = []
                
                # Add all positive polygons
                for pos_poly in positive_polygons:
                    vertices.extend(pos_poly)
                    codes.extend([Path.MOVETO] + [Path.LINETO] * (len(pos_poly) - 1))
                
                # Add all holes
                for hole_poly in hole_polygons:
                    vertices.extend(hole_poly)
                    codes.extend([Path.MOVETO] + [Path.LINETO] * (len(hole_poly) - 1))
                
                # Create path and patch with proper fill rule for holes
                path = Path(vertices, codes)
                patch = PathPatch(path, facecolor=color, alpha=0.4, edgecolor=color, linewidth=2)
                patch.set_path_effects([])  # Ensure proper hole rendering
                ax.add_patch(patch)
            else:
                # No holes, simple filled polygons
                for pos_poly in positive_polygons:
                    ax.add_patch(MplPolygon(pos_poly, closed=True, fill=True, color=color, alpha=0.4))
                    ax.add_patch(MplPolygon(pos_poly, closed=True, fill=False, edgecolor=color, linewidth=2))
        else:
            # Only holes present (unusual case) - show as dashed outlines
            for hole_poly in hole_polygons:
                ax.add_patch(MplPolygon(hole_poly, closed=True, fill=False, edgecolor=color, linewidth=2, linestyle='--'))

    def visualize_annotations_masked(self, image: np.ndarray, annotation_ids: Union[int, List[int]], 
                                   ax: Optional[plt.Axes] = None, show: bool = True) -> Optional[plt.Axes]:
        """
        Visualize COCO annotations by masking out background pixels using RLE masks,
        correctly handling holes by subtracting hole masks from positive masks.
        All background pixels will be set to 0 for each object.
        Uses create_segmentation_mask for consistent mask generation.
        
        Args:
            image (np.ndarray): Image array to visualize on.
            annotation_ids (Union[int, List[int]]): Annotation ID(s) to visualize.
            ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates new figure.
            show (bool): Whether to show the plot immediately.
            
        Returns:
            Optional[plt.Axes]: The axes object if ax was None, otherwise None.
        """
        # Create axes if not provided
        return_ax = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(15, 15))
        
        # Handle single annotation ID case
        if isinstance(annotation_ids, int):
            annotation_ids = [annotation_ids]
            
        try:
            anns = self.coco.loadAnns(annotation_ids)
        except KeyError:
            print("No annotations found for the given IDs.")
            if return_ax:
                plt.close()
            return None
            
        if not anns:
            print("No annotations found for the given IDs.")
            if return_ax:
                plt.close()
            return None

        img_info = self.coco.imgs[anns[0]['image_id']]
        img_disp = image.copy()
        img_height = img_info["height"]
        img_width = img_info["width"]

        for ann in anns:
            segmentation = ann.get('segmentation', None)
            segmentation_types = ann.get('segmentation_types', None) 
            bbox = ann.get('bbox', None)

            if segmentation and bbox:
                x, y, w, h = map(int, bbox)
                
                # Use create_segmentation_mask to handle positive areas and holes
                full_mask = create_segmentation_mask(
                    segmentation,
                    segmentation_types,
                    img_height,
                    img_width
                )
                
                if full_mask is not None:
                    # Crop mask to bbox
                    mask = full_mask[y:y+h, x:x+w]
                    obj_img = torch.from_numpy(img_disp[y:y+h, x:x+w]).float()
                    
                    # If image is multi-channel, expand mask dims for broadcasting
                    if obj_img.dim() == 3 and mask.dim() == 2:
                        mask = mask.unsqueeze(-1)
                    
                    # Set background to 0
                    obj_img = obj_img * mask
                    
                    # Place masked object back into image for visualization
                    img_disp[y:y+h, x:x+w] = obj_img.numpy().astype(img_disp.dtype)

        ax.imshow(img_disp)
        ax.set_title("Masked objects (using create_segmentation_mask)")
        ax.axis('off')
        
        if show and return_ax:
            plt.show()
            
        return ax if return_ax else None
