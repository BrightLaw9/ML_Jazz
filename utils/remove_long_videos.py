import subprocess
import glob
import os


def get_duration(file):
    cmd = f'ffprobe -v error -i "{file}" -show_entries format=duration -v quiet -of csv="p=0"'
    out = subprocess.check_output(
        cmd,
        shell=True,
        stderr=subprocess.STDOUT
        )
    return float(out)

for file in glob.glob("./train_wav_files/**", recursive=True):
    if not os.path.isfile(file):
        continue
    if get_duration(file) >= 20*60:
        print(file)
        os.remove(file)
