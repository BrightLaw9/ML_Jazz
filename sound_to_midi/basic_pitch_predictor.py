"""This file uses Spotify's audio intelligence BasicPitch tool to convert audio to MIDI"""

import sys
import os
import glob
import pathlib

from basic_pitch.inference import predict_and_save, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
existing_files = glob.glob(f".\\train_midi_files\\*")
existing_set = set([file.split("\\")[-1] for file in existing_files]) # Allow for O(1) access for duplicate checking, compared to O(n) for list

for file_path in glob.glob(f".\\train_wav_files\\split_{sys.argv[1]}\\*"): 
    if file_path.split("\\")[-1].replace(".wav", "_basic_pitch.mid") in existing_set: 
        f = file_path.split("\\")[-1].replace(".wav", "_basic_pitch.mid")
        print(f"**Skipping {f}, all ready exists**")
        continue
    predict_and_save(
        [pathlib.Path(file_path)], 
        pathlib.Path("./train_midi_files"), 
        True, 
        False, 
        False, 
        False, 
        basic_pitch_model
    )
