# -*- coding: utf-8 -*-

import sys

import librosa

import glob

from monophonic import wave_to_midi

# Run this script in the ML_Jazz directory to automatically convert wav to midi for training
def run():
    print("Starting...")
    print(f"Processing for ./train_wav_files/split_{sys.argv[1]}/*")
    all_midi_files = glob.glob("./train_midi_files/*")
    for file_name in glob.glob(f"./train_wav_files/split_{sys.argv[1]}/*"): 
        file_name = file_name.replace("\\", "/")
        if (file_name.replace(".wav", ".midi").split("/")[-1] not in [file.replace("\\", "/").split("/")[-1] for file in all_midi_files]): 
            print(file_name)
            file_in = file_name#sys.argv[1]
            file_out = f"./train_midi_files/{file_in.split('/')[-1].replace('.wav', '')}.midi"
            audio_data, srate = librosa.load(file_in, sr=None)
            print("Audio file loaded!")
            midi = wave_to_midi(audio_data, srate=srate)
            print("Conversion finished!")
            with open(file_out, 'wb') as file:
                midi.writeFile(file)
            print(f"Done {file_name}") 
    
    print("Exiting!")


if __name__ == "__main__":
    run()