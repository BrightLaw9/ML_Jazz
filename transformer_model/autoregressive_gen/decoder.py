import pretty_midi
from dataclasses import dataclass

@dataclass
class Event:
    event_type: str
    start: int
    pitch: int
    velocity: int = 0

    def __str__(self) -> str:
        if self.event_type in ("note_on", "note_off"):
            note = pretty_midi.note_number_to_name(self.pitch)
            if self.event_type == "note_on":
                return (
                    f"NoteOn({note}, t={self.start}ms, vel={self.velocity})"
                )
            else:
                return (
                    f"NoteOff({note}, t={self.start}ms)"
                )

        if self.event_type == "time_shift":
            return f"TimeShift(+{self.start}ms)"

        if self.event_type == "velocity":
            return f"Velocity({self.velocity})"

        return f"Event(type={self.event_type})"

NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = 128
TIME_SHIFT_OFFSET = 256

MAX_TIME_SHIFT_MS = 1000
TIME_RES_MS = 5
MAX_TIME_TOKENS = MAX_TIME_SHIFT_MS // TIME_RES_MS  # 200

VELOCITY_OFFSET = TIME_SHIFT_OFFSET + MAX_TIME_TOKENS  # 456


def token_to_event(token: int) -> Event:
    # NOTE ON
    if NOTE_ON_OFFSET <= token < NOTE_OFF_OFFSET:
        return Event(
            event_type="note_on",
            start=0,
            pitch=token - NOTE_ON_OFFSET,
            velocity=0,
        )

    # NOTE OFF
    if NOTE_OFF_OFFSET <= token < TIME_SHIFT_OFFSET:
        return Event(
            event_type="note_off",
            start=0,
            pitch=token - NOTE_OFF_OFFSET,
            velocity=0,
        )

    # TIME SHIFT
    if TIME_SHIFT_OFFSET <= token < VELOCITY_OFFSET:
        shift_steps = token - TIME_SHIFT_OFFSET + 1
        return Event(
            event_type="time_shift",
            start=shift_steps * TIME_RES_MS,
            pitch=0,
            velocity=0,
        )

    # VELOCITY
    if token >= VELOCITY_OFFSET:
        return Event(
            event_type="velocity",
            start=0,
            pitch=0,
            velocity=token - VELOCITY_OFFSET,
        )

    raise ValueError(f"Unknown token: {token}")
