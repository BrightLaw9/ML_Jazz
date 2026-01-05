import pretty_midi
from dataclasses import dataclass
from ..constants import *

# @dataclass
# class Event:
#     event_type: str
#     start: int
#     pitch: int
#     velocity: int = 0

#     def __str__(self) -> str:
#         if self.event_type in ("note_on", "note_off"):
#             note = pretty_midi.note_number_to_name(self.pitch)
#             if self.event_type == "note_on":
#                 return (
#                     f"NoteOn({note}, t={self.start}ms, vel={self.velocity})"
#                 )
#             else:
#                 return (
#                     f"NoteOff({note}, t={self.start}ms)"
#                 )

#         if self.event_type == "time_shift":
#             return f"TimeShift(+{self.start}ms)"

#         if self.event_type == "velocity":
#             return f"Velocity({self.velocity})"

#         return f"Event(type={self.event_type})"


def token_to_event(token: int):
    # NOTE ON
    if NOTE_ON_OFFSET <= token < DUR_OFFSET:
        note = pretty_midi.note_number_to_name(token)
        return (f"Tok: {token}, Note: {note}")

    
    elif DUR_OFFSET <= token < TIME_SHIFT_OFFSET:
        dur = (token - DUR_OFFSET) * DUR_RES_MS
        return (f"Duration: {dur}")
    
    elif TIME_SHIFT_OFFSET <= token < VELOCITY_OFFSET:
        shift = (token - TIME_SHIFT_OFFSET) * TIME_RES_MS
        return (f"Time shift: {shift}")
    
    elif VELOCITY_OFFSET <= token:
        vel = bin_to_velocity(token - VELOCITY_OFFSET)
        return (f"Velocity: {vel}")


    raise ValueError(f"Unknown token: {token}")
