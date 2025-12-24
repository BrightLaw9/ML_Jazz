import pretty_midi
import torch
import math
from dataclasses import dataclass

# Token offsets
NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = 128
TIME_SHIFT_OFFSET = 256

MAX_TIME_SHIFT_MS = 1000
TIME_RES_MS = 5
MAX_TIME_TOKENS = MAX_TIME_SHIFT_MS // TIME_RES_MS  # 100

VELOCITY_OFFSET = TIME_SHIFT_OFFSET + MAX_TIME_TOKENS #456

@dataclass
class Event:
    event_type : str
    start : int
    pitch : int
    velocity : int = 0

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
        notes.extend(inst.notes)

    # # Sort notes by start time
    # notes.sort(key=lambda n: n.start)

    note_events = []

    for note in notes:
        note_events.append(Event(
            event_type="note_on",
            start=note.start,
            pitch=note.pitch,
            velocity=note.velocity))
        note_events.append(Event(
            event_type="note_off",
            start=note.end,
            pitch=note.pitch))
    
    note_events.sort(key=lambda e : (e.start, e.event_type == "note_off")) # first look at time, then if same time, first turn note off

    for note in note_events:
        # Time shift
        delta = note.start - current_time
        if delta > 0:
            ms = int(delta * 1000)
            while ms > 0:
                shift = min(ms, MAX_TIME_SHIFT_MS)
                events.append(
                    TIME_SHIFT_OFFSET + (shift // TIME_RES_MS)
                )
                ms -= shift
            current_time = note.start

        if (note.event_type == "note_on"):
            vel_bin = velocity_to_bin(note.velocity)
            events.append(VELOCITY_OFFSET + vel_bin)

            events.append(NOTE_ON_OFFSET + note.pitch)
        elif (note.event_type == "note_off"):
            events.append(NOTE_OFF_OFFSET + note.pitch)

    return torch.tensor(events, dtype=torch.long)

