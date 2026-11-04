"""Private native gesture-preview commands; no supplied classes or page commands."""

from typing import Literal

from shared.desktop_protocol import Counter, SessionId, WireModel


class NativeGestureCommand(WireModel):
    version: Literal[1]
    session_id: SessionId
    command_id: Counter
    camera_epoch: Counter
    based_on_sequence: Counter
    kind: Literal[
        "open_camera",
        "enable",
        "choose_owner",
        "stop",
        "close_camera",
        "surface_ack",
        "heartbeat",
        "shutdown",
    ]
