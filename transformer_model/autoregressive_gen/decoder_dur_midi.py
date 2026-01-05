import pretty_midi
import torch
from ..constants import *

pm = pretty_midi.PrettyMIDI()

piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

#MIDI_PATH = "train_midi_pt_dur/Jazz MIDI 3/AutumnLeaves.pt"
#MIDI_PATH = "train_midi_pt_dur/JAZZ MIDIS ABC/ABC/au_privave-CParker_ss3.pt"\
#MIDI_PATH = "transformer_gen/sample_2026-01-05_00-49-15-550491.pt"
MIDI_PATH = "transformer_gen/sample_2026-01-05_01-39-53-999098.pt"
seq = torch.load(MIDI_PATH)

events = seq.tolist()[0]

current_time = 0.0
i = 0

while i < len(events):
    if TIME_SHIFT_OFFSET <= events[i] < VELOCITY_OFFSET:
        current_time += TIME_RES_MS * (events[i] - TIME_SHIFT_OFFSET)
        i += 1
        if (current_time > 120 * 1000 and current_time < 140 * 1000):
            print("Cur time", current_time)
    elif events[i] >= VELOCITY_OFFSET:
        if (i >= len(events) - 3):
            break
        vel = bin_to_velocity(events[i] - VELOCITY_OFFSET)
        i += 1
        pitch = events[i]
        pitch_name = pretty_midi.note_number_to_name(pitch)
        i += 1
        dur = (events[i] - DUR_OFFSET) * DUR_RES_MS
        i += 1
        if (current_time > 120 * 1000 and current_time < 140 * 1000):
            print(f"Vel: {vel}, Pitch: {pitch}, Pitch Name: {pitch_name}, Dur: {dur}")
        piano.notes.append(
            pretty_midi.Note(
                velocity=vel,
                pitch=pitch,
                start=(current_time) / 1000,
                end=(current_time + dur) / 1000
            )
        )

pm.instruments.append(piano)

#pm.write('output.mid')

#pm.write('blues_for_alice_3.mid')
pm.write('autumn_leaves.mid')
