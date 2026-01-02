@echo off
setlocal enabledelayedexpansion

set INPUT_FILE=file_list.txt
set FILES=

for /f "usebackq delims=" %%F in ("%INPUT_FILE%") do (
    set FILES=!FILES! ".\output_wavs\new_piano\%%F"
)

echo !FILES!
done
