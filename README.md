# ML_Jazz
Using machine learning (Tensorflow and Magenta) to create jazz music

The ultimate goal of this project is to have AI generate a comprehensible and melodical piece of music. It should resemble the jazz style (swinging, note articulations, dynamics).

Sample data, collected as wav files (some scraped from the web, others I personally recorded!), was converted to MIDI files for training. The Python files in sound_to_midi achieve this conversion programmatically and quickly. Also, it is programmed to produce monophonic MIDI files, so a cleaner melody line can be heared. Past trials with many channels resulted in a rather distorted sound. 

Using Magenta's Melody RNN ML Model (built on Tensorflow), I trained the model with the MIDI files to produce somewhat decent sounding jazz.

The generated melody is then added over a backing rhythm section to complete the jazz experience.

# Installation note
Python 3.7 64 bit had to be used in order to run pip install magenta and install sucessfully. The Numba and Python-rtmidi libraries created incompatibility and build errors when attempting to use versions higher than 3.7. 