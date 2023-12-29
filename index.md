# Some background:
- Jazz is a free-flowing musical language that is rooted in swing, and improvised melodies.
- In this project, I aimed to have the AI generate a comprehensible and melodical piece of music, with contrasting ranges in pitch, jazz articulations, and dynamics.
- An Attention Recurrent Neural Network (RNN) was used to capture longer patterns within the music, allowing for better musical phrases and context. 
- The song was based on the Charlie Parker tune, Blues for Alice. Piano was selected to be the lead instrument.
- The model was trained on collected recordings on the web of pianists, as well as snippets which I personally recorded!
- The training workflow was as such: Recorded WAV files -> MIDI Files -> NoteSequences (serializing midi into tfrecord for training) -> SequenceExamples (labelled training data to feed into neural network) -> Training the RNN (hyperparameter tuning as well!) & Evaluate training -> Generate Music (with latest stored model checkpoints - the weights, biases, and other information of network)

Take a listen to some of the generated music below!

| Sample 1 | <audio src="Blues_for_Alice_ML_v1.mp3" controls></audio>
Notes & Notable Timestamps: 
- Trained on a 2 layer RNN with 64 processing units each
- 

| Sample 2 | <audio src="Blues_for_Alice_ML_v2_swing.mp3" controls></audio>
Notes & Notable Timestamps: 
- Trained on a 2 layer RNN with 128 processing units each
- At 1:20, there is instance of overfitting
- 1:32 - interesting swing groove happening! (solid chrous of improv)
- 2:02 - fit the Abm7 -> Db7 harmony nicely
- Occasional jazz articulations/phrases appearing (one at 2:37 - 2:38)

| Sample 3 | <audio src="Blues_for_Alice_ML_v3_color.mp3" controls></audio>
Notes & Notable Timestamps: 
- Trained on a 3 layer RNN with 64 processing units each
- 1:20 - Interesting stepping up sequence
- The stepping up sequence transitions from mid to high range at 1:30 - 1:33
- 1:33 - 1:34 - Melodical phrase generated!
- 1:40 - 2:36 - A period of exotic back and forths with the upper and lower ranges of pitch (effects of attention)
- 2:36 - Return to natural phrasing
