"""Private NDJSON child for the native built-in presenter. No camera/OS input."""

import os
import sys

from backend.desktop import decode_command, serve
from backend.presentation import PresentationSession
from shared.presentation_protocol import PresentationCommand


def main():
    session = PresentationSession()
    try:
        with (
            os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0) as source,
            os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0) as output,
        ):
            serve(
                source,
                output,
                session=session,
                decoder=lambda line: decode_command(line, PresentationCommand),
            )
    finally:
        session.close()
        session.dispatcher.wait_released(timeout=0.3)


if __name__ == "__main__":
    main()
