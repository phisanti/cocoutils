#!/usr/bin/env python3
"""
Example script demonstrating the cocoutils health check module.

This script shows various ways to use the health checker programmatically.
"""

from cocoutils.health import CocoHealthChecker
from cocoutils.health.formatters import HumanFormatter, TokenOptimizedFormatter


def example_basic_usage():
    """Basic health check example."""
    print("=" * 80)
    print("Example 1: Basic Health Check")
    print("=" * 80)
    
    checker = CocoHealthChecker("test.json")
    results = checker.run_all_checks()
    
    # Access individual results
    stats = results['basic_stats']
    print(f"\nDataset contains:")
    print(f"  - {stats['n_images']} images")
    print(f"  - {stats['n_annotations']} annotations")
    print(f"  - {stats['n_categories']} categories")
    
    # Check for issues
    if not results['category_validation']['valid']:
        print(f"\nWARNING: Found {results['category_validation']['invalid_count']} invalid category IDs")
    
    if not results['orphaned_annotations']['valid']:
        print(f"\nWARNING: Found {results['orphaned_annotations']['count']} orphaned annotations")
    
    if not results['bbox_validation']['valid']:
        print(f"\nWARNING: Found {results['bbox_validation']['count']} invalid bounding boxes")
    
    print()


def example_human_formatter():
    """Example using human-readable formatter."""
    print("=" * 80)
    print("Example 2: Human-Readable Format")
    print("=" * 80)
    
    checker = CocoHealthChecker("test.json")
    results = checker.run_all_checks()
    
    formatter = HumanFormatter(results, "test.json")
    output = formatter.format()
    print(output)


def example_token_optimized_formatter():
    """Example using token-optimized formatter."""
    print("=" * 80)
    print("Example 3: Token-Optimized Format (for AI agents)")
    print("=" * 80)
    
    checker = CocoHealthChecker("test.json")
    results = checker.run_all_checks()
    
    formatter = TokenOptimizedFormatter(results, "test.json")
    output = formatter.format()
    print(output)
    print()


def example_category_distribution():
    """Example analyzing category distribution."""
    print("=" * 80)
    print("Example 4: Category Distribution Analysis")
    print("=" * 80)
    
    checker = CocoHealthChecker("test.json")
    results = checker.run_all_checks()
    
    cat_dist = results['category_distribution']
    print("\nAnnotations per category:")
    
    # Sort by count (descending)
    sorted_cats = sorted(
        cat_dist.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    for cat_id, info in sorted_cats:
        print(f"  {info['name']:20s} ({cat_id:2d}): {info['count']:5d} annotations")
    
    # Calculate percentages
    total = sum(info['count'] for info in cat_dist.values())
    print(f"\nTotal: {total} annotations")
    
    print("\nPercentage breakdown:")
    for cat_id, info in sorted_cats:
        percentage = (info['count'] / total) * 100
        print(f"  {info['name']:20s}: {percentage:5.1f}%")
    
    print()


def example_conditional_processing():
    """Example with conditional processing based on validation results."""
    print("=" * 80)
    print("Example 5: Conditional Processing")
    print("=" * 80)
    
    checker = CocoHealthChecker("test.json")
    results = checker.run_all_checks()
    
    # Define critical checks
    critical_checks = [
        ('category_validation', "Invalid category IDs detected"),
        ('orphaned_annotations', "Orphaned annotations detected"),
    ]
    
    warnings = []
    errors = []
    
    for check_name, message in critical_checks:
        if not results[check_name]['valid']:
            errors.append(message)
    
    # Check for warnings (non-critical)
    if results['bbox_validation']['count'] > 0:
        warnings.append(f"Found {results['bbox_validation']['count']} invalid bounding boxes")
    
    if results['images_without_annotations']['count'] > 0:
        warnings.append(f"Found {results['images_without_annotations']['count']} images without annotations")
    
    # Report results
    print("\nValidation Results:")
    
    if errors:
        print("\nERRORS (must fix):")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\nWARNINGS (should review):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("  All checks passed!")
    
    # Decision making
    if errors:
        print("\nRecommendation: Fix errors before using this dataset")
        return False
    elif warnings:
        print("\nRecommendation: Review warnings, but dataset is usable")
        return True
    else:
        print("\nRecommendation: Dataset is ready to use")
        return True


def example_verbose_mode():
    """Example using verbose mode for detailed analysis."""
    print("=" * 80)
    print("Example 6: Verbose Mode (Per-Image Analysis)")
    print("=" * 80)
    
    checker = CocoHealthChecker("test.json")
    results = checker.run_all_checks(verbose=True)
    
    # Analyze per-image statistics
    image_stats = results['per_image_stats']
    
    print(f"\nAnalyzing {len(image_stats)} images...")
    
    # Find images with most annotations
    sorted_images = sorted(
        image_stats.items(),
        key=lambda x: x[1]['ann_count'],
        reverse=True
    )
    
    print("\nTop 5 images by annotation count:")
    for img_id, info in sorted_images[:5]:
        print(f"  {info['file_name']:40s}: {info['ann_count']:4d} annotations, "
              f"{info['multipolygon_count']:2d} multipolygons")
    
    # Find images with multipolygons
    mp_images = [(img_id, info) for img_id, info in image_stats.items() 
                 if info['multipolygon_count'] > 0]
    
    if mp_images:
        print(f"\nImages with multipolygons ({len(mp_images)}):")
        for img_id, info in mp_images[:5]:
            print(f"  {info['file_name']:40s}: {info['multipolygon_count']:2d} multipolygons")
    
    print()


if __name__ == "__main__":
    # Run all examples
    examples = [
        example_basic_usage,
        example_human_formatter,
        example_token_optimized_formatter,
        example_category_distribution,
        example_conditional_processing,
        example_verbose_mode,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}\n")
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
