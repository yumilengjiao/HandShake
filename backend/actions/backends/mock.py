"""Deterministic in-memory faults, including late side effects after stop."""
from threading import Event, Lock

from backend.actions.backends.dry_run import DryRunBackend
from shared.contracts import Action, ActionResult, Capability


class MockBackend(DryRunBackend):
    def __init__(self, *, fault=None, block=None, capabilities=tuple(Capability), release_failure=False):
        super().__init__(capabilities)
        if fault not in {None, "before", "after"}:
            raise ValueError("fault must be before/after/None")
        self.fault, self.block, self.release_failure = fault, block, release_failure
        self.entered = Event()
        self.held = set()
        self.release_calls = 0
        self._lock = Lock()

    def execute(self, command, now_ns):
        self.entered.set()
        if self.block is not None:
            self.block.wait()
        if self.fault == "before":
            raise RuntimeError("mock failure before side effect")
        with self._lock:
            self.commands.append(command)
            if command.action == Action.BUTTON_DOWN:
                self.held.add("primary")
            elif command.action == Action.BUTTON_UP:
                self.held.discard("primary")
        if self.fault == "after":
            raise RuntimeError("mock failure after side effect")
        return ActionResult(command_id=command.command_id, accepted=True, injection_status="not_attempted",
                            completed_at_ns=now_ns, acknowledgement_level="mock", timebase=command.timebase)

    def release_all(self):
        with self._lock:
            self.release_calls += 1
            if self.release_failure:
                raise RuntimeError("mock release failure")
            self.held.clear()
        return True
