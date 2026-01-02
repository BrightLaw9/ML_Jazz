@echo off
REM Set the model directory
set MODEL_DIR=C:\Users\lawre\ML_Jazz\maestro_checkpoint\train

REM Set the input folder
set INPUT_FOLDER=.\output_wavs\new_piano

REM Initialize an empty variable to hold all .wav files
set WAV_FILES=

REM Loop over all .wav files and append to WAV_FILES
for %%F in ("%INPUT_FOLDER%\*.wav") do (
    set WAV_FILES=!WAV_FILES! "%%F"
)

REM Enable delayed variable expansion
setlocal enabledelayedexpansion

REM Run the Python script once with all .wav files
python ..\magenta\magenta\models\onsets_frames_transcription\onsets_frames_transcription_transcribe.py --model_dir="%MODEL_DIR%" !WAV_FILES!

echo Done.
pause
