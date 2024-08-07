# ML_Jazz
Using machine learning (Tensorflow and Magenta) to create jazz music

> [!NOTE]
> **Check out the website for more information (including generated music): https://brightlaw9.github.io/ML_Jazz/**

The ultimate goal of this project is to have AI generate a comprehensible and melodical piece of music. It should resemble the jazz style (swinging, note articulations, dynamics). In the end, elements of jazz have been learned and integrated into the music, but no where near as natural and comforting as a human would do so. 

Sample data, collected as wav files (some scraped from the web, others I personally recorded!), was converted to MIDI files for training. The Python files in ```sound_to_midi``` achieve this conversion programmatically and quickly. Also, it is programmed to produce monophonic MIDI files, so a cleaner melody line can be heared. Past trials with many channels resulted in a rather distorted sound. 

Using Magenta's Melody RNN ML Model (built on Tensorflow), I trained the model with the MIDI files to produce somewhat decent sounding jazz. Files used for training are written to ```./training/seq_examples``` (this directory was git ignored as the files were large). These SequenceExamples are Magenta's ways of generating inputs and labels to be fed into the machine learning model. 

The workflow can be seen as follows: 
  1) Recorded WAV files (in ```./train_wav_files``` - not all was included due to size constraints)
  2) To MIDI Files (```./train_midi_files```)
  3) Convert to NoteSequences (serializing midi into tfrecord for training) (steps below can be run using ```.\machine_learn.bat```)
  4) Create SequenceExamples (labelled training data to feed into neural network)
  5) Training the RNN (hyperparameter tuning as well!) & Evaluate training
  6) Generate Music (with latest stored model checkpoints - the weights, biases, and other information of network)

The generated melody is then mixed over a backing rhythm section to complete the jazz experience.

Note the train_midi_files/ and train_wav_files directories have been git ignored due to the large space taken up due to training samples (approx. 1000 audio samples)

Spotify's audio-to-midi engine (trained neural network predicting midi) basic-pitch and ffmpeg were libraries used to prepare audio data for training.

# Installation note
Python 3.7 64 bit had to be used in order to run pip install magenta and install sucessfully. The Numba and Python-rtmidi libraries created incompatibility and build errors when attempting to use versions higher than 3.7. 
