import os
import argparse
import subprocess
from pathlib import Path

def normalize_audio(dataset_path: str, target_sr: int = 44100):
    base_path = Path(dataset_path)
    if not base_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Find all drum audio stems
    drum_stems = []
    for ext in ["*.ogg", "*.wav", "*.mp3"]:
        for file in base_path.rglob(ext):
            if file.stem.lower() == "drums":
                drum_stems.append(file)
                
    if not drum_stems:
        print(f"No drum stems found in {dataset_path}.")
        return

    print(f"Found {len(drum_stems)} drum files. Normalizing to {target_sr}Hz...")

    success_count = 0
    fail_count = 0

    for drum_file in drum_stems:
        # Create a temp file path
        temp_file = drum_file.with_name(f"temp_{drum_file.name}")
        
        # FFmpeg command to normalize sample rate and ensuring mono/stereo consistency (let's stick to stereo 2ch)
        cmd = [
            "ffmpeg",
            "-y",                # Overwrite output
            "-i", str(drum_file),
            "-ar", str(target_sr), # Target Sample Rate
            "-ac", "2",          # Target 2 Channels (Stereo)
            "-loglevel", "error", # Keep it quiet
            str(temp_file)
        ]
        
        try:
            print(f"Processing: {drum_file.parent.name}/{drum_file.name}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace the original file with the new normalized temp file
                temp_file.replace(drum_file)
                success_count += 1
            else:
                print(f"✖ Failed on {drum_file.name}: {result.stderr}")
                if temp_file.exists():
                    temp_file.unlink() # Cleanup
                fail_count += 1
                
        except FileNotFoundError:
            print("ERROR: 'ffmpeg' command not found in PATH.")
            print("Please ensure FFmpeg is installed and added to your system PATH.")
            if temp_file.exists(): temp_file.unlink()
            return
        except Exception as e:
            print(f"Unexpected error on {drum_file.name}: {e}")
            if temp_file.exists(): temp_file.unlink()
            fail_count += 1

    print("\n--- Normalization Summary ---")
    print(f"Successfully normalized: {success_count}/{len(drum_stems)}")
    print(f"Failed: {fail_count}/{len(drum_stems)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize all drums.ogg/wav to a specific sample rate (default 44100Hz).")
    parser.add_argument("dataset_path", help="Path to your dataset directory.")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate (e.g. 44100, 48000).")
    args = parser.parse_args()

    normalize_audio(args.dataset_path, target_sr=args.sr)
