"""
Formatters for health check results.

Provides formatting for both human-readable and token-optimized output.
"""

from typing import Dict, Any


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class HealthCheckFormatter:
    """Base formatter for health check results."""
    
    def __init__(self, results: Dict[str, Any], filepath: str):
        """
        Initialize formatter.
        
        Args:
            results (Dict[str, Any]): Health check results.
            filepath (str): Path to the COCO file being checked.
        """
        self.results = results
        self.filepath = filepath
    
    def format(self) -> str:
        """Format the results. To be implemented by subclasses."""
        raise NotImplementedError


class HumanFormatter(HealthCheckFormatter):
    """Human-friendly formatter with clear sections and visual hierarchy."""
    
    def format(self) -> str:
        """
        Format results for human readability.
        
        Returns:
            str: Formatted health check report.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"COCO Health Check: {self.filepath}")
        lines.append("=" * 80)
        
        # Basic statistics
        lines.append("\nBasic Statistics:")
        stats = self.results['basic_stats']
        lines.append(f"  Images:      {stats['n_images']}")
        lines.append(f"  Annotations: {stats['n_annotations']}")
        lines.append(f"  Categories:  {stats['n_categories']}")
        
        # Orphaned categories
        lines.append("\nOrphaned Category Check:")
        orphaned_cats = self.results['orphaned_categories']
        if orphaned_cats['valid']:
            lines.append(f"  {Colors.GREEN}OK{Colors.RESET} - All categories are properly used")
        else:
            if orphaned_cats['unused_count'] > 0:
                lines.append(f"  {Colors.YELLOW}WARNING{Colors.RESET} - Found {orphaned_cats['unused_count']} unused categories (defined but never used):")
                for cat_id, name in orphaned_cats['unused_categories'][:10]:
                    lines.append(f"    - Category {cat_id} ({name})")
                if orphaned_cats['unused_count'] > 10:
                    lines.append(f"    ... and {orphaned_cats['unused_count'] - 10} more")
            
            if orphaned_cats['undefined_count'] > 0:
                lines.append(f"  {Colors.RED}ERROR{Colors.RESET} - Found {orphaned_cats['undefined_count']} undefined categories (used but not defined):")
                for cat_id in list(orphaned_cats['undefined_categories'])[:10]:
                    lines.append(f"    - Category ID {cat_id}")
                if orphaned_cats['undefined_count'] > 10:
                    lines.append(f"    ... and {orphaned_cats['undefined_count'] - 10} more")
        
        # Category distribution
        lines.append("\nAnnotations per Class:")
        cat_dist = self.results['category_distribution']
        if cat_dist:
            for cat_id in sorted(cat_dist.keys()):
                info = cat_dist[cat_id]
                lines.append(f"  Class {cat_id} ({info['name']}): {info['count']} annotations")
        else:
            lines.append(f"  {Colors.YELLOW}WARNING{Colors.RESET} - No categories found")
        
        # Multipolygon statistics
        lines.append("\nMultipolygon Annotations:")
        mp_stats = self.results['multipolygon_stats']
        lines.append(f"  Total: {mp_stats['count']}")
        if mp_stats['count'] > 0 and 'per_image_stats' in self.results:
            lines.append("  Details (first 10):")
            for ann_id, img_id, poly_count in mp_stats['details'][:10]:
                lines.append(f"    - Annotation {ann_id} (image {img_id}): {poly_count} polygons")
        
        # Orphaned annotations
        lines.append("\nOrphaned Annotation Check:")
        orphaned = self.results['orphaned_annotations']
        if orphaned['valid']:
            lines.append(f"  {Colors.GREEN}OK{Colors.RESET} - No orphaned annotations found")
        else:
            lines.append(f"  {Colors.RED}ERROR{Colors.RESET} - Found {orphaned['count']} orphaned annotations:")
            for ann_id in orphaned['orphaned_ids'][:10]:
                lines.append(f"    - Annotation {ann_id}")
            if orphaned['count'] > 10:
                lines.append(f"    ... and {orphaned['count'] - 10} more")
        
        # Images without annotations
        lines.append("\nImages Without Annotations:")
        no_ann = self.results['images_without_annotations']
        lines.append(f"  Count: {no_ann['count']}")
        if no_ann['count'] > 0 and 'per_image_stats' in self.results:
            lines.append("  Examples (first 10):")
            for img_id, filename in no_ann['images'][:10]:
                lines.append(f"    - Image {img_id}: {filename}")
        
        # Bounding box validation
        lines.append("\nBounding Box Validation:")
        bbox_val = self.results['bbox_validation']
        if bbox_val['valid']:
            lines.append(f"  {Colors.GREEN}OK{Colors.RESET} - All bounding boxes are valid")
        else:
            lines.append(f"  {Colors.RED}ERROR{Colors.RESET} - Found {bbox_val['count']} invalid bounding boxes:")
            for ann_id, reason in bbox_val['invalid_bboxes'][:10]:
                lines.append(f"    - Annotation {ann_id}: {reason}")
            if bbox_val['count'] > 10:
                lines.append(f"    ... and {bbox_val['count'] - 10} more")
        
        # Per-image statistics
        lines.append("\nPer-Image Statistics:")
        if 'per_image_stats' in self.results:
            image_info = self.results['per_image_stats']
            lines.append(f"  {'Image ID':<12} {'Filename':<40} {'Annotations':<15} {'Multipolygons'}")
            lines.append(f"  {'-'*12} {'-'*40} {'-'*15} {'-'*12}")
            for img_id in sorted(image_info.keys())[:20]:
                info = image_info[img_id]
                filename = info['file_name'][:38] + ".." if len(info['file_name']) > 40 else info['file_name']
                lines.append(f"  {img_id:<12} {filename:<40} {info['ann_count']:<15} {info['multipolygon_count']}")
            if len(image_info) > 20:
                lines.append(f"  ... and {len(image_info) - 20} more images")
        else:
            summary = self.results['per_image_summary']
            if 'annotations' in summary:
                ann = summary['annotations']
                lines.append(f"  Annotations per image - min: {ann['min']}, max: {ann['max']}, avg: {ann['avg']:.1f}")
            if 'multipolygons' in summary:
                mp = summary['multipolygons']
                lines.append(f"  Multipolygons per image - min: {mp['min']}, max: {mp['max']}, avg: {mp['avg']:.1f}")
        
        lines.append("\n" + "=" * 80)
        lines.append("Health check complete")
        lines.append("=" * 80)
        
        return "\n".join(lines)


class TokenOptimizedFormatter(HealthCheckFormatter):
    """Token-optimized formatter for agent consumption."""
    
    def format(self) -> str:
        """
        Format results with minimal tokens for agent processing.
        
        Returns:
            str: Compact formatted report.
        """
        lines = []
        lines.append(f"COCO_HEALTH_CHECK: {self.filepath}")
        lines.append("")
        
        # Basic stats
        stats = self.results['basic_stats']
        lines.append(f"STATS: images={stats['n_images']} annotations={stats['n_annotations']} categories={stats['n_categories']}")
        
        # Orphaned categories
        orphaned_cats = self.results['orphaned_categories']
        if orphaned_cats['valid']:
            lines.append("ORPHANED_CATEGORIES: OK")
        else:
            status_parts = []
            if orphaned_cats['unused_count'] > 0:
                status_parts.append(f"unused={orphaned_cats['unused_count']}")
                unused_list = [f"{cat_id}({name})" for cat_id, name in orphaned_cats['unused_categories'][:5]]
                lines.append(f"ORPHANED_CATEGORIES: WARNING {' '.join(status_parts)}")
                lines.append(f"  unused: {', '.join(unused_list)}")
            if orphaned_cats['undefined_count'] > 0:
                status_parts.append(f"undefined={orphaned_cats['undefined_count']}")
                undefined_list = [str(cat_id) for cat_id in list(orphaned_cats['undefined_categories'])[:5]]
                if orphaned_cats['unused_count'] == 0:
                    lines.append(f"ORPHANED_CATEGORIES: ERROR {' '.join(status_parts)}")
                lines.append(f"  undefined: {', '.join(undefined_list)}")
        
        # Category distribution
        cat_dist = self.results['category_distribution']
        if cat_dist:
            dist_str = " | ".join([f"{cat_id}({info['name']})={info['count']}" 
                                   for cat_id, info in sorted(cat_dist.items())])
            lines.append(f"CLASS_DISTRIBUTION: {dist_str}")
        
        # Multipolygons
        mp_stats = self.results['multipolygon_stats']
        lines.append(f"MULTIPOLYGONS: count={mp_stats['count']}")
        
        # Orphaned annotations
        orphaned = self.results['orphaned_annotations']
        if orphaned['valid']:
            lines.append("ORPHANED_ANNOTATIONS: OK")
        else:
            lines.append(f"ORPHANED_ANNOTATIONS: ERROR count={orphaned['count']}")
            if orphaned['orphaned_ids']:
                lines.append(f"  ids: {', '.join(map(str, orphaned['orphaned_ids'][:10]))}")
        
        # Images without annotations
        no_ann = self.results['images_without_annotations']
        lines.append(f"IMAGES_WITHOUT_ANNOTATIONS: count={no_ann['count']}")
        
        # Bounding boxes
        bbox_val = self.results['bbox_validation']
        if bbox_val['valid']:
            lines.append("BBOXES: OK")
        else:
            lines.append(f"BBOXES: ERROR count={bbox_val['count']}")
            if bbox_val['invalid_bboxes']:
                bbox_list = [f"{ann_id}({reason})" for ann_id, reason in bbox_val['invalid_bboxes'][:5]]
                lines.append(f"  invalid: {', '.join(bbox_list)}")
        
        # Per-image summary
        if 'per_image_summary' in self.results:
            summary = self.results['per_image_summary']
            if 'annotations' in summary:
                ann = summary['annotations']
                lines.append(f"PER_IMAGE_ANNOTATIONS: min={ann['min']} max={ann['max']} avg={ann['avg']:.1f}")
            if 'multipolygons' in summary:
                mp = summary['multipolygons']
                lines.append(f"PER_IMAGE_MULTIPOLYGONS: min={mp['min']} max={mp['max']} avg={mp['avg']:.1f}")
        
        return "\n".join(lines)
