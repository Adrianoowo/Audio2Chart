import os

class ReverseParser:
    def __init__(self, fps=50, bpm=120.0):
        """
        ReverseParser maps the AI probability matrices back into Clone Hero compatible .chart format.
        """
        self.fps = fps
        self.ms_per_frame = 1000.0 / self.fps
        self.resolution = 192
        self.bpm = bpm
        
        # 1 beat = 60,000ms / BPM
        # 1 tick = (1 beat duration) / resolution
        self.ms_per_tick = (60000.0 / self.bpm) / self.resolution

    def export_chart(self, matrix, output_dir, threshold=0.5):
        """
        matrix: numpy array of shape [num_frames, 5] containing values 0.0 -> 1.0 (after sigmoid)
        """
        events = []
        for frame_idx in range(matrix.shape[0]):
            ms = frame_idx * self.ms_per_frame
            tick = int(round(ms / self.ms_per_tick))
            
            for lane in range(5):
                # Trigger a drum hit if the AI prediction passes the threshold limit
                if matrix[frame_idx, lane] > threshold:
                    events.append({"tick": tick, "lane": lane})
                    
        # Sort and mildly sanitize bounds
        events = sorted(events, key=lambda x: (x["tick"], x["lane"]))
        
        unique_events = []
        last_ticks = {-1: -9999, 0: -9999, 1: -9999, 2: -9999, 3: -9999, 4: -9999}
        
        for e in events:
            # Prune out machine-gun hits (e.g. AI hitting same lane twice in <40ms)
            if e["tick"] - last_ticks[e["lane"]] > (40 / self.ms_per_tick):
                unique_events.append(e)
                last_ticks[e["lane"]] = e["tick"]
        
        chart_path = os.path.join(output_dir, "notes.chart")
        with open(chart_path, "w", encoding="utf-8") as f:
            f.write("[Song]\n{\n")
            f.write(f"  Resolution = {self.resolution}\n")
            f.write("}\n")
            
            f.write("[SyncTrack]\n{\n")
            f.write("  0 = TS 4\n")
            f.write(f"  0 = B {int(self.bpm * 1000)}\n")
            f.write("}\n")
            
            f.write("[Events]\n{\n}\n")
            
            f.write("[ExpertDrums]\n{\n")
            for e in unique_events:
                # Strict Clone Hero mapping N <lane> <length_0> (we drop sustains!)
                f.write(f"  {e['tick']} = N {e['lane']} 0\n")
            f.write("}\n")
            
        print(f"[*] Map Generated: {len(unique_events)} AI predicted drum strikes mapped.")
        
    def export_ini(self, output_dir, artist="AI", title="Generated Drum Track"):
        """ Generate essential metadata requirements for the Songs directory """
        ini_path = os.path.join(output_dir, "song.ini")
        with open(ini_path, "w", encoding="utf-8") as f:
            f.write("[song]\n")
            f.write(f"name={title}\n")
            f.write(f"artist={artist}\n")
            f.write("album=Audio2Chart AI\n")
            f.write("genre=Generative AI\n")
            f.write("year=2026\n")
            f.write("diff_drums=4\n")
            f.write("multiplier_note=116\n")
            f.write("pro_drums=0\n")
            f.write("icon=ai\n")
