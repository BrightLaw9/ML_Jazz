# -*- coding: utf-8 -*-

import sys

import librosa

from monophonic import wave_to_midi

def run():
    print("Starting...")
    file_in = sys.argv[1]
    file_out = f"./midi_files/{file_in.split('/')[-1].replace('.wav', '')}.midi"
    audio_data, srate = librosa.load(file_in, sr=None)
    print("Audio file loaded!")
    midi = wave_to_midi(audio_data, srate=srate)
    print("Conversion finished!")
    with open (file_out, 'wb') as file:
        midi.writeFile(file)
    print("Done. Exiting!")


if __name__ == "__main__":
    run()