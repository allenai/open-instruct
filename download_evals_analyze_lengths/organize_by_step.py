#!/usr/bin/env python3
"""
Script to organize directories by step number.

This script takes a path and reorganizes subdirectories that contain step information
(e.g., "_step_700") by creating intermediate directories for each step and moving
the corresponding directories into them.

Example:
    Before: /parent/lmeval-..._step_700-on-aime...
    After:  /parent/step_700/lmeval-..._step_700-on-aime...
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def extract_step_number(dirname):
    """
    Extract step number from directory name.
    
    Args:
        dirname: Directory name to extract step from
        
    Returns:
        Step number as string if found, None otherwise
    """
    # Pattern to match _step_XXX where XXX is a number
    match = re.search(r'_step_(\d+)', dirname)
    if match:
        return match.group(1)
    return None


def organize_directories(base_path, dry_run=False):
    """
    Organize directories by step number.
    
    Args:
        base_path: Path to the directory containing subdirectories to organize
        dry_run: If True, only print what would be done without making changes
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist")
        return
    
    if not base_path.is_dir():
        print(f"Error: Path {base_path} is not a directory")
        return
    
    # Group directories by step number
    step_groups = defaultdict(list)
    
    # Scan all items in the base directory
    for item in os.listdir(base_path):
        item_path = base_path / item
        
        # Only process directories
        if not item_path.is_dir():
            continue
        
        # Skip directories that are already step_XXX directories
        if re.match(r'^step_\d+$', item):
            print(f"Skipping existing step directory: {item}")
            continue
        
        # Extract step number
        step_num = extract_step_number(item)
        if step_num:
            step_groups[step_num].append(item)
    
    if not step_groups:
        print("No directories with step numbers found")
        return
    
    # Print summary
    print(f"\nFound {len(step_groups)} unique step(s):")
    for step_num in sorted(step_groups.keys(), key=int):
        print(f"  step_{step_num}: {len(step_groups[step_num])} directories")
    
    # Organize directories
    print("\nOrganizing directories...")
    for step_num in sorted(step_groups.keys(), key=int):
        step_dir_name = f"step_{step_num}"
        step_dir_path = base_path / step_dir_name
        
        if dry_run:
            print(f"\n[DRY RUN] Would create directory: {step_dir_path}")
        else:
            # Create step directory if it doesn't exist
            step_dir_path.mkdir(exist_ok=True)
            print(f"\nCreated/verified directory: {step_dir_path}")
        
        # Move each directory into the step directory
        for dir_name in step_groups[step_num]:
            src_path = base_path / dir_name
            dst_path = step_dir_path / dir_name
            
            if dry_run:
                print(f"  [DRY RUN] Would move: {dir_name}")
                print(f"            to: {step_dir_name}/{dir_name}")
            else:
                try:
                    shutil.move(str(src_path), str(dst_path))
                    print(f"  Moved: {dir_name}")
                except Exception as e:
                    print(f"  Error moving {dir_name}: {e}")
    
    if dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to apply changes.")
    else:
        print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Organize directories by step number",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python organize_by_step.py /path/to/base/directory
  python organize_by_step.py /path/to/base/directory --dry-run
        """
    )
    parser.add_argument(
        "path",
        help="Path to the directory containing subdirectories to organize"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making any changes"
    )
    
    args = parser.parse_args()
    
    organize_directories(args.path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

