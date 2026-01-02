#!/usr/bin/env python

from __future__ import print_function

import csv
import glob
import json
import math
import os
import shlex
import subprocess
from optparse import OptionParser
from pathlib import Path

INPUT_DIR = "output_wavs/new_piano"
INPUT_ROOT = Path(INPUT_DIR)
OUTPUT_DIR = Path("output_wavs/new_piano")

def get_video_length(filename):
    output = subprocess.check_output(("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                      "default=noprint_wrappers=1:nokey=1", filename)).strip()
    video_length = int(float(output))
    print("Video length in seconds: " + str(video_length))

    return video_length


def ceildiv(a, b):
    return int(math.ceil(a / float(b)))


def split_by_seconds(filename, split_length, acodec="copy",
                     extra="", video_length=None, **kwargs):
    if split_length and split_length <= 0:
        print("Split length can't be 0")
        raise SystemExit

    if not video_length:
        video_length = get_video_length(filename)
    split_count = ceildiv(video_length, split_length)
    if split_count == 1:
        print("Video length is less then the target split length.")
        return
        #raise SystemExit

    split_cmd = ["ffmpeg", "-i", filename, "-acodec", acodec] + shlex.split(extra)
    try:
        path = Path(filename)

        filebase = path.stem
        fileext = path.suffix

        relative = path.parent.relative_to(INPUT_ROOT)
        out_base_path = OUTPUT_DIR / relative
        out_base_path.mkdir(parents=True, exist_ok=True)
    except IndexError as e:
        raise IndexError("No . in filename. Error: " + str(e))
    for n in range(0, split_count):
        split_args = []
        if n == 0:
            split_start = 0
        else:
            split_start = split_length * n

        split_args += ["-ss", str(split_start), "-t", str(split_length),
                       str(out_base_path / (f"{filebase}" + "-" + str(n + 1) + "-of-" +
                       str(split_count) + fileext))]
        print("About to run: " + " ".join(split_cmd + split_args))
        subprocess.check_output(split_cmd + split_args)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for file in glob.glob(f"./{INPUT_DIR}/**", recursive=True): 
        if os.path.isfile(file):
            split_by_seconds(file, 3*60)


if __name__ == '__main__':
    main()