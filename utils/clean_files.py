"""Use this script to clean files that aren't wav files which were scraped by YouTube-DL"""

import os
import glob

def clean(): 
    for file in glob.glob("./train_wav_files/*"): 
        if not file.endswith(".wav"): 
            os.remove(file)

if __name__ == "__main__": 
    clean()
