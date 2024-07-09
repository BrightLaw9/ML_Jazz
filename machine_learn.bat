
REM run in ML_Jazz directory
set INPUT_DIRECTORY="./train_midi_files"
set SEQUENCES_TFRECORD="./training/note_sequences.tfrecord"

set SEQUENCE_EXAMPLE_DIR=./training/seq_examples/
set MODEL_CHECKPOINTS_DIR="./training/basic_pitch_melody_rnn_ckpts/"
REM set MODEL_CHECKPOINTS_DIR="./training/melody_rnn_ckpts_64_3_layer/" 
REM set MODEL_CHECKPOINTS_DIR="./training/melody_rnn_ckpts_128/"
set GENERATED_OUTPUT_DIR="./generated_music"

convert_dir_to_note_sequences --input_dir=%INPUT_DIRECTORY% --output_file=%SEQUENCES_TFRECORD% --recursive

melody_rnn_create_dataset --config=attention_rnn --input=%SEQUENCES_TFRECORD% --output_dir=%SEQUENCE_EXAMPLE_DIR% --eval_ratio=0.10

melody_rnn_train --config=attention_rnn --run_dir=%MODEL_CHECKPOINTS_DIR% --sequence_example_file=%SEQUENCE_EXAMPLE_DIR%training_melodies.tfrecord --hparams="batch_size=64,rnn_layer_sizes=[64,64]" --num_training_steps=20000
REM melody_rnn_train --config=attention_rnn --run_dir=%MODEL_CHECKPOINTS_DIR% --sequence_example_file=%SEQUENCE_EXAMPLE_DIR%training_melodies.tfrecord --hparams="batch_size=128,rnn_layer_sizes=[128,128]" --num_training_steps=20000
REM melody_rnn_train --config=attention_rnn --run_dir=%MODEL_CHECKPOINTS_DIR% --sequence_example_file=%SEQUENCE_EXAMPLE_DIR%training_melodies.tfrecord --hparams="batch_size=64,rnn_layer_sizes=[64,64,64]" --num_training_steps=20000

melody_rnn_generate --config=attention_rnn --run_dir=%MODEL_CHECKPOINTS_DIR% --output_dir=%GENERATED_OUTPUT_DIR% --num_outputs=3 --num_steps=1024 --hparams="batch_size=64,rnn_layer_sizes=[64,64]" --primer_midi="./train_midi_files/Blues_for_allice_head.midi"
REM melody_rnn_generate --config=attention_rnn --run_dir=%MODEL_CHECKPOINTS_DIR% --output_dir=%GENERATED_OUTPUT_DIR% --num_outputs=3 --num_steps=1024 --hparams="batch_size=128,rnn_layer_sizes=[128,128]" --primer_midi="./midi_files/Blues_for_allice_head.midi"
REM melody_rnn_generate --config=attention_rnn --run_dir=%MODEL_CHECKPOINTS_DIR% --output_dir=%GENERATED_OUTPUT_DIR% --num_outputs=3 --num_steps=1024 --hparams="batch_size=64,rnn_layer_sizes=[64,64,64]" --primer_midi="./midi_files/Blues_for_allice_head.midi"
