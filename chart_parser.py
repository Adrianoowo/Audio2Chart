import numpy as np

class ChartParser:
    def __init__(self, fps=50):
        """
        fps determines the time-step resolution.
        fps=50 means each frame represents 20ms of audio (1000ms / 50).
        """
        self.fps = fps
        self.ms_per_frame = 1000.0 / self.fps
        
    def parse_file(self, filepath):
        """
        Parses a .chart file, strictly looking for [SyncTrack] and [ExpertDrums].
        Returns flat timestamped events arrays and bpm boundaries.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            
        resolution = 192
        sync_track = []
        events_drums = []
        
        current_section = ""
        for line in lines:
            if line.startswith("["):
                current_section = line
                continue
            
            if current_section == "[Song]":
                if line.startswith("Resolution"):
                    parts = line.split("=")
                    if len(parts) == 2:
                        resolution = int(parts[1].strip())
                        
            elif current_section == "[SyncTrack]":
                if "=" in line:
                    parts = line.split("=")
                    tick = int(parts[0].strip())
                    event = parts[1].strip().split()
                    if event[0] == "B":
                        # BPM is standardized as BPM * 1000 in .chart files
                        bpm = int(event[1]) / 1000.0
                        sync_track.append({"tick": tick, "bpm": bpm})
                        
            elif current_section == "[ExpertDrums]":
                if "=" in line:
                    parts = line.split("=")
                    tick = int(parts[0].strip())
                    event = parts[1].strip().split()
                    # Event format: N lane length -> e.g. N 0 0
                    if event[0] == "N":
                        lane = int(event[1])
                        # Clone Hero drum lanes: Kick 0, Red 1, Yellow 2, Blue 3, Green 4
                        # Note: 'length' is intentionally ignored since drums do not have sustains.
                        if 0 <= lane <= 4:
                            events_drums.append({"tick": tick, "lane": lane})
                            

        # Compute absolute true time (in ms) for each SyncTrack BPM marker
        sync_track = sorted(sync_track, key=lambda x: x["tick"])
        
        bpm_map = []
        current_ms_time = 0.0
        
        # Inject standard default values if sync track forgets absolute zero marker
        if not sync_track or sync_track[0]["tick"] != 0:
            bpm_map.append({"tick": 0, "bpm": 120.0, "start_ms": 0.0})
        
        for st in sync_track:
            tick = st["tick"]
            bpm = st["bpm"]
            
            if bpm_map:
                prev = bpm_map[-1]
                delta_ticks = tick - prev["tick"]
                beats = delta_ticks / float(resolution)
                ms = (beats * 60000.0) / prev["bpm"]
                current_ms_time = prev["start_ms"] + ms
                
            bpm_map.append({"tick": tick, "bpm": bpm, "start_ms": current_ms_time})
            
        def get_ms_for_tick(target_tick):
            # Find the active BPM boundary for the tick
            active_bpm = bpm_map[0]
            for b in bpm_map:
                if b["tick"] <= target_tick:
                    active_bpm = b
                else:
                    break
            
            delta_ticks = target_tick - active_bpm["tick"]
            beats = delta_ticks / float(resolution)
            ms = (beats * 60000.0) / active_bpm["bpm"]
            return active_bpm["start_ms"] + ms
            
        # Transform abstract logic ticks to real audio millisecond onsets
        for event in events_drums:
            event["ms"] = get_ms_for_tick(event["tick"])
            
        return events_drums, bpm_map
        
    def create_matrix(self, events_drums, max_time_ms):
        """
        Converts the list of millisecond-timed drum hits into a quantized
        binary one-hot matrix for neural network ingestion.
        Shape: [num_frames, 5] representing (Kick, Red, Yellow, Blue, Green)
        """
        num_frames = int(np.ceil(max_time_ms / self.ms_per_frame))
        
        matrix = np.zeros((num_frames, 5), dtype=np.float32)
        
        for event in events_drums:
            ms = event["ms"]
            lane = event["lane"]
            
            frame_idx = int(ms / self.ms_per_frame)
            if 0 <= frame_idx < num_frames:
                matrix[frame_idx, lane] = 1.0
                
        return matrix

if __name__ == "__main__":
    parser = ChartParser(fps=50) # 20ms windows
    print("ChartParser initialized. Ready to quantize drum charts into matrices.")
