import os
import shutil
import argparse
from pathlib import Path

def prune_dataset(dataset_path: str, dry_run: bool = True):
    base_path = Path(dataset_path)
    if not base_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    directories = [d for d in base_path.iterdir() if d.is_dir()]
    print(f"Found {len(directories)} song folders to evaluate.")

    kept = 0
    removed = 0
    removed_dirs = []

    for folder in directories:
        has_chart = len(list(folder.glob("*.chart"))) > 0
        
        # Checking for any common drum stem audio formats 
        # Typically drums.ogg, but checking wav/mp3 as well just in case
        has_drums = False
        drum_stems = ["drums.ogg", "drums.wav", "drums.mp3"]
        for ds in drum_stems:
            if (folder / ds).exists():
                has_drums = True
                break

        if has_chart and has_drums:
            kept += 1
            # Clear out excess files (like guitar.ogg, rhythm.ogg, etc.) to save space?
            # We will just prune the invalid folders for now and keep valid ones exactly as they are.
        else:
            removed += 1
            removed_dirs.append(folder)

    print(f"\n--- Pruning Summary ---")
    print(f"Valid Folders (Kept): {kept}")
    print(f"Invalid Folders: {removed}")
    
    if dry_run:
        print("\n[DRY RUN] The following folders WOULD be deleted:")
        for r in removed_dirs:
            print(f" - {r.name} (Missing: {'drums stem' if not has_drums else ''} {'chart file' if not has_chart else ''})")
        print("\nTo actually delete these folders, run with the --execute flag.")
    else:
        print("\nDeleting invalid folders...")
        for r in removed_dirs:
            try:
                shutil.rmtree(r)
                print(f"Deleted: {r.name}")
            except Exception as e:
                print(f"Failed to delete {r.name}: {e}")
        print("Deletion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune dataset to folders containing only valid .chart and drum stems.")
    parser.add_argument("dataset_path", help="Path to your dataset directory.")
    parser.add_argument("--execute", action="store_true", help="Perform the actual deletion (destructive).")
    args = parser.parse_args()

    prune_dataset(args.dataset_path, dry_run=not args.execute)
