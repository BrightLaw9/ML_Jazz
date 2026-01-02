import pretty_midi
import torch
import math
from ..constants import *

def velocity_to_bin(vel, bins=32):
    return min(bins - 1, vel * bins // 128)

def midi_to_events(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)

    events = []

    current_time = 0.0

    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        if "bass" in inst.name.lower():
            continue
        notes.extend(inst.notes)

    # # Sort notes by start time
    # notes.sort(key=lambda n: n.start)

    notes.sort(key=lambda e : (e.start, e.end)) # first look at time, then if same time, first turn note off

    for note in notes:
        # Time shift
        delta = note.start - current_time
        if delta > 0:
            ms = int(delta * 1000)
            while ms >= TIME_RES_MS:
                shift = min(ms, MAX_TIME_SHIFT_MS)
                events.append(
                    TIME_SHIFT_OFFSET + (shift // TIME_RES_MS)
                )
                ms -= shift
            current_time = note.start

        vel_bin = velocity_to_bin(note.velocity)
        events.append(VELOCITY_OFFSET + vel_bin)

        events.append(NOTE_ON_OFFSET + note.pitch)

        delta_note = note.end - note.start
        ms = int(delta_note * 1000)
        #print("Ms ", ms)
        events.append(
            DUR_OFFSET + (min(ms, MAX_DUR_SHIFT_MS) // DUR_RES_MS)
        )

    # print("Events", events)



    return torch.tensor(events, dtype=torch.long)
