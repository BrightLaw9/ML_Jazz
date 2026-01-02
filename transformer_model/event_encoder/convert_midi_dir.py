#from .midi_to_events import midi_to_events
from .midi_to_events_dur import midi_to_events
import os
from pathlib import Path
import torch

def convert_midi_dir(midi_root, out_root):
    os.makedirs(out_root, exist_ok=True)

    for path in list(Path(midi_root).rglob("*.mid")):

        try:
            seq = midi_to_events(path)
            root = Path(midi_root)
            out_root = Path(out_root)

            relative = path.relative_to(root)
            out_path = out_root / relative.with_suffix(".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(seq, out_path)
            print(f"Saved {out_path} ({len(seq)} tokens)")
        except Exception as e:
            print(f"Skipping {path}: {e}")

#convert_midi_dir("output_wavs", "train_midi_pt_3")

convert_midi_dir("train_midi_files_new", "train_midi_pt_dur")
