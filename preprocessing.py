import librosa
import librosa.feature
import torch
import pickle
import numpy as np
import tempfile
from pathlib import Path
from chart_parser import ChartParser
import argparse

def process_dataset(dataset_path: str, output_pkl: str = "data.pkl"):
    """
    Rethought Preprocessing script for Clone Hero Drums.
    Instead of Malody formats, dynamically utilizes our chart_parser 
    and slices the `drums.ogg` mel-spectrograms mapping them to the neural matrices.
    """
    parser = ChartParser(fps=50) # 20 ms frames
    mel_hop_length = 512
    sr = 44100
    
    # 87 frames of mel-spectrogram correspond to ~ 1.0 sec of audio
    # For a given 20ms matrix timestep, we will provide the surrounding ~1 sec context
    mel_context_frames = 87 
    half_context = mel_context_frames // 2
    
    base_path = Path(dataset_path)
    if not base_path.exists():
        print("Dataset path not found.")
        return

    data_store = {}
    
    # Recursively find all folders holding a .chart or .mid file across nested subdirectories
    chart_folders = [f.parent for f in base_path.rglob("*.chart")]
    mid_folders = [f.parent for f in base_path.rglob("*.mid")]
    song_folders = list(set(chart_folders + mid_folders))
    
    print(f"Discovered {len(song_folders)} folders recursively. Compiling training data...")
    
    valid_count = 0
    chunk_count = 0
    
    # Establish disk-streaming bounds strictly onto the volatile Windows Temp drive
    tensors_dir = Path(tempfile.gettempdir()) / "audio2chart_tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, folder in enumerate(song_folders):
        try:
            chart_file = None
            is_midi = False
            
            # Look for natively mapped Phase 2 targets OR Phase 1 midis natively
            found_chart = list(folder.glob("*.chart"))
            if found_chart:
                chart_file = found_chart[0]
            else:
                found_mid = list(folder.glob("*.mid"))
                if found_mid:
                    chart_file = found_mid[0]
                    is_midi = True

            drum_files = list(folder.glob("drums*.ogg")) + list(folder.glob("drums*.wav"))

            if not drum_files or chart_file is None:
                continue

            print(f"Extracting Mel-Spectrogram ({idx+1}/{len(song_folders)}): {folder.name}")
            
            y_mixed = None
            for d_file in drum_files:
                y_part, _ = librosa.load(d_file, sr=sr, mono=True)
                if y_mixed is None:
                    y_mixed = y_part
                else:
                    max_len = max(len(y_mixed), len(y_part))
                    if len(y_mixed) < max_len:
                        y_mixed = np.pad(y_mixed, (0, max_len - len(y_mixed)))
                    if len(y_part) < max_len:
                        y_part = np.pad(y_part, (0, max_len - len(y_part)))
                    y_mixed += y_part
            
            y = y_mixed
            
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmin=20, fmax=22050, hop_length=mel_hop_length
            )
            mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
            
            # Utilize the correct bindings
            if is_midi:
                events_drums, _ = parser.parse_midi(str(chart_file))
            else:
                events_drums, _ = parser.parse_file(str(chart_file))
                
            audio_duration_ms = (len(y) / sr) * 1000
            label_matrix = parser.create_matrix(events_drums, max_time_ms=audio_duration_ms)
            
            # Generate the inputs [timestep, 1, 128, 87] tracking the label logic windows
            # Align Matrix frames with Mel-spectrogram frames
            num_timesteps = label_matrix.shape[0]
            
            # We don't want to kill RAM, so let's compute and chunk the dataset into small sequences.
            # E.g., sequences of 200 timesteps (~ 4 seconds each).
            seq_len = 200
            
            for seq_start in range(0, num_timesteps - seq_len, seq_len):
                seq_mels = []
                seq_labels = []
                
                for t in range(seq_start, seq_start + seq_len):
                    # time in seconds
                    current_s = (t * parser.ms_per_frame) / 1000.0
                    center_mel_frame = librosa.time_to_frames(times=current_s, sr=sr, hop_length=mel_hop_length)
                    
                    start_f = center_mel_frame - half_context
                    end_f = center_mel_frame + (mel_context_frames - half_context)
                    
                    # Padding zero handlers
                    if start_f < 0:
                        slice_mel = torch.cat((torch.zeros([128, -start_f]), mel_spectrogram[:, :end_f]), dim=1)
                    elif end_f > mel_spectrogram.shape[1]:
                        slice_mel = torch.cat((mel_spectrogram[:, start_f:], torch.zeros([128, end_f - mel_spectrogram.shape[1]])), dim=1)
                    else:
                        slice_mel = mel_spectrogram[:, start_f:end_f]
                        
                    seq_mels.append(slice_mel.unsqueeze(0)) # Shape [1, 128, 87]
                    seq_labels.append(torch.tensor(label_matrix[t]))
                    
                # Serialize the sequence dynamically to Disk avoiding dictionary bloat
                tensor_payload = {
                    "input": torch.stack(seq_mels), # [seq_len, 1, 128, 87]
                    "label": torch.stack(seq_labels) # [seq_len, 5]
                }
                torch.save(tensor_payload, tensors_dir / f"seq_{idx}_{seq_start}.pt")
                chunk_count += 1
                
            valid_count += 1

        except Exception as e:
            print(f"Skipping {folder.name} due to unexpected parsing error: {e}")

    print(f"Compiled {chunk_count} total sliding windows from {valid_count} songs.")
    print("Preprocessing complete. Results saved securely into /dataset_tensors/")

if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("dataset_path", help="Folder containing validated chart folders.")
    args = cli.parse_args()
    process_dataset(args.dataset_path)
