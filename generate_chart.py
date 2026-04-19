import os
import argparse
import subprocess
import shutil
import torch
import librosa
from pathlib import Path
from model import ChartNet
from reverse_parser import ReverseParser

def prepare_audio_features(drums_path, sr=44100, mel_context_frames=87):
    """ Loads target track and computes matching sliding-window mel-spectrogram chunks. """
    y, _ = librosa.load(drums_path, sr=sr, mono=True)
    hop_length = 512
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmin=20, fmax=22050, hop_length=hop_length
    )
    mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

    total_frames = int((len(y) / sr) / 0.02) # Map entirely to 20ms resolutions (fps=50)
    
    half_context = mel_context_frames // 2
    inputs = []
    
    for t in range(total_frames):
        current_s = t * 0.02
        center_f = librosa.time_to_frames(times=current_s, sr=sr, hop_length=hop_length)
        start_f = center_f - half_context
        end_f = center_f + (mel_context_frames - half_context)
        
        # Zero-pad the edges if window overlaps bounds
        if start_f < 0:
            slice_mel = torch.cat((torch.zeros([128, -start_f]), mel_spectrogram[:, :end_f]), dim=1)
        elif end_f > mel_spectrogram.shape[1]:
            slice_mel = torch.cat((mel_spectrogram[:, start_f:], torch.zeros([128, end_f - mel_spectrogram.shape[1]])), dim=1)
        else:
            slice_mel = mel_spectrogram[:, start_f:end_f]
            
        inputs.append(slice_mel.unsqueeze(0)) # Yield [1, 128, 87]
        
    inputs = torch.stack(inputs).unsqueeze(0) # Batched wrapper shape [1, seq_len, 1, 128, 87]
    return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Path to the custom mp3/ogg file.")
    parser.add_argument("--model", default="checkpoints/drum_model_epoch_30.pth", help="Trained PT weights.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold to yield drum hits.")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Fatal: Could not pinpoint absolute audio target > {audio_path}")
        return

    output_dir = Path("CloneHero_Songs") / audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pipeline Action: Demucs integration
    print("[1/4] Spawning Demucs Subprocess (Isolating Drums Stem)...")
    subprocess.run(["python", "-m", "demucs", "--two-stems=drums", "-n", "htdemucs", str(audio_path)])
    
    demucs_drum_path = Path(f"separated/htdemucs/{audio_path.stem}/drums.wav")
    
    if not demucs_drum_path.exists():
        print(f"Warning: Demucs output failed or diverted. Falling back to non-isolated core audio {audio_path.name}")
        demucs_drum_path = audio_path
        
    
    # 2. Pipeline Action: Audio Mel Features extraction
    print("[2/4] Transcribing Mel-Spectrogram Windows...")
    inputs = prepare_audio_features(demucs_drum_path)
    
    # 3. Pipeline Action: Launching Inference Net
    print("[3/4] Activating Clone Hero Bi-LSTM Predictor (AI inference)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChartNet(output_dim=5).to(device)
    
    if Path(args.model).exists():
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(" -> Weights mapped successfully.")
    else:
        print(f" -> WARNING: Untrained Network! The file {args.model} doesn't exist.")
        
    model.eval()
    inputs = inputs.to(device)
    
    with torch.no_grad():
        logits = model(inputs) # Output dense sequence tensor
        
        # Squeeze through sigmoid to break unbounded linear logits strictly mapped within 0.0->1.0 probability
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
    # 4. Pipeline Action: Translating back to CH Chart format
    print("[4/4] Writing game logic constraints...")
    rp = ReverseParser()
    rp.export_chart(probs, str(output_dir), threshold=args.threshold)
    rp.export_ini(str(output_dir), title=audio_path.stem)
    
    # Move source track into the playable folder alongside our map/ini
    target_bgm = output_dir / "song.ogg"
    shutil.copy(audio_path, target_bgm)
    
    print(f"\n==================== SUCCESS ====================")
    print(f"Map Compiled! Drag '{output_dir}' into Clone Hero's Songs Folder.")

if __name__ == "__main__":
    main()
