import os
import argparse
import subprocess
from pathlib import Path

def convert_sng_to_chart(dataset_path: str):
    base_path = Path(dataset_path)
    if not base_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Scan for both .sng strings and .mid (MIDI) binaries which compose the majority of RB datasets
    target_files = list(base_path.rglob("*.sng")) + list(base_path.rglob("*.mid"))
    if not target_files:
        print(f"No .sng or .mid files found in {dataset_path}.")
        return

    print(f"Found {len(target_files)} unconverted charts. Starting Onyx conversion...")

    success_count = 0
    fail_count = 0

    for sng_file in target_files:
        # Assuming onyx works like: onyx convert "input.sng"
        # We don't have the exact CLI documentation from the user, but this is typical.
        try:
            print(f"Converting {sng_file.name}...")
            # Using the explicit path to onyx.exe provided by the user
            onyx_path = r"C:\Users\adema\Documents\Audio2Chart\onyx-command-line-20251011-windows-x64\onyx.exe"
            result = subprocess.run(
                [onyx_path, "build", "--target", "ch", str(sng_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✔ Success: {sng_file.name}")
                success_count += 1
                # Delete original .sng file if conversion successful? Let's leave it for safety for now.
            else:
                print(f"✖ Failed: {sng_file.name}")
                print(result.stderr)
                fail_count += 1
                
        except FileNotFoundError:
            print("ERROR: 'onyx' command not found in PATH.")
            print("Please ensure Onyx is installed and added to your system PATH.")
            return
        except Exception as e:
            print(f"Unexpected error when converting {sng_file.name}: {e}")
            fail_count += 1

    print("\n--- Conversion Summary ---")
    print(f"Successfully converted: {success_count}/{len(target_files)}")
    print(f"Failed: {fail_count}/{len(target_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert .sng files to .chart using Onyx.")
    parser.add_argument("dataset_path", help="Path to your dataset directory containing the songs.")
    args = parser.parse_args()

    convert_sng_to_chart(args.dataset_path)
