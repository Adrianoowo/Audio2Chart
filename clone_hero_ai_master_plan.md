# ARCHITECTURE & EXECUTION PLAN: CLONE HERO DRUM CHART AI

## 0. CONTEXT FOR THE AI ASSISTANT
If you are an AI reading this, your job is to act as the Senior AI Architect and Lead Python Developer for this project. The human user is driving the vision but their Python skills are basic. You will write the code, debug the inevitable dimension mismatch errors, and guide them step-by-step. 

**The Ultimate Goal:** A complete pipeline that takes a raw `.mp3`/`.ogg` file, uses Demucs to isolate the drum stem, feeds that stem into a trained sequence-to-sequence neural network, and spits out a perfectly timed, playable `.chart` file for Clone Hero on Expert Drums.

**The Current State:** - We are starting with **Expert Drums ONLY**. No guitar, no sustains, no hammer-ons.
- The user already has a pristine dataset of ~3000 official Rock Band/Guitar Hero multitracks (audio stems + `.chart` files).
- We are NOT building the model from absolute scratch. We are forking/adapting an existing open-source project called `audio2chart` (by 3podi), which currently only maps guitar, and modifying it to map drum hits. It is the project currently open.

---

## 1. PHASE 1: DATASET PREPARATION & SANITIZATION
We have the files, but they are messy. The AI needs to write scripts to clean this shit up.

1.  **File Conversion:** Run Onyx to batch convert any lingering `.sng` files in the 3000-song dataset into standard `.chart` files.
2.  **Dataset Pruning:** Write a Python script using `pathlib` to iterate through the massive `charts/` directory. Keep only the directories that contain both a valid `.chart` file and a `drums.ogg` (or equivalent drum stem). Discard the rest.
3.  **Stem Normalization:** Ensure all `drums.ogg` files are the same sample rate (e.g., 44.1kHz). Write a script using `ffmpeg` or `librosa` to standardize them if necessary.

---

## 2. PHASE 2: THE PARSER (THE HARDEST SCRIPT)
The neural network cannot read a `.chart` file directly. You need to write a Python parser to translate the text files into mathematical arrays. 

1.  **Parsing `[SyncTrack]`:** The AI must read the BPM and time signature changes, mapping the chart's abstract "ticks" into exact milliseconds. If a chart has floating BPMs, the script needs to calculate the absolute timestamp for every beat.
2.  **Parsing `[ExpertDrums]`:** Extract every drum hit. 
    * Standard mapping usually involves 5 lanes: Kick (0), Red (1), Yellow (2), Blue (3), Green (4).
    * Since drums do not have sustains in Clone Hero, ignore any length values attached to the notes. Focus purely on the onset timestamps.
3.  **Tokenization / Matrix Creation:** Convert the song into discrete time-steps (e.g., every 10ms-20ms). For every time-step, output a binary array `[Kick, Red, Yellow, Blue, Green]`. A `1` means a transient hit, a `0` means silence.

---

## 3. PHASE 3: HIJACKING `audio2chart`
We are adapting 3podi's `audio2chart` repository.

1.  **Clone the Repo:** Fork the codebase. This is already done as you are working on the forked repo.
2.  **Modify the Tokenizer:** `audio2chart` uses an Encodec model and an autoregressive Transformer, originally built for 5 frets + open notes + sustains. You must rip out the sustain logic entirely. Rewrite the target vocabulary to strictly represent the drum pad combinations. 
3.  **Feature Extraction:** Ensure the Encodec layers or Mel-spectrogram generators are tuned to pick up low-frequency transients (like the kick drum) and high-frequency crashes (cymbals), which differ heavily from a guitar's frequency range.

---

## 4. PHASE 4: TRAINING THE MODEL

1.  **Dataloader Setup:** Write a custom PyTorch `Dataset` class that pairs the raw `drums.ogg` audio arrays with the tokenized binary matrices we generated in Phase 2.
2.  **Loss Function:** Standard Cross-Entropy loss for multi-label classification.
3.  **Training Loop:** Run the epochs until the loss stops decreasing. Periodically test the model's predictions on a validation set of songs it hasn't seen to make sure it isn't just memorizing charts.

---

## 5. PHASE 5: INFERENCE & THE FINAL PIPELINE
Once the model is trained and weights are saved, build the actual user-facing tool.

1.  **The Wrapper Script:** A command-line tool where the user inputs `python generate_chart.py my_new_song.mp3`.
2.  **Demucs Integration:** The script programmatically calls Demucs to split `my_new_song.mp3` and isolates `drums.wav`.
3.  **AI Prediction:** Feed `drums.wav` into our trained model. The model spits out the raw token arrays.
4.  **Reverse Parser (The Exporter):** Write a script that translates those raw arrays back into Clone Hero `.chart` text format. It needs to calculate a generic `[SyncTrack]` (or attempt to run a beat-detection algorithm to map the BPM) and format the `[ExpertDrums]` notes.
5.  **Packaging:** Move the original audio, the `.chart` file, and an `ini` file into a neat folder ready to be dragged into the Clone Hero `Songs` directory.
