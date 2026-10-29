"""T17/T18 host with an independent clock watchdog; no camera/UI/model calls."""
from threading import Event, Thread
import time

from backend.actions.dispatcher import Dispatcher
from backend.actions.mapper import ActionMapper
from backend.interaction.engine import InteractionEngine


class InteractionRuntime:
    def __init__(self, controller, *, model_version, backend=None, config=None, clock=None, policy=None, watchdog=True):
        self.clock = clock or time.monotonic_ns
        self.controller = controller
        self.dispatcher = Dispatcher(controller, backend=backend, config=config, clock=self.clock)
        self.mapper = ActionMapper(config)
        self.engine = InteractionEngine(controller, policy=policy, model_version=model_version,
                                        stop_handler=self.dispatcher.stop,
                                        invalidate_handler=lambda now: self.dispatcher.stop(now, suspend=True))
        self._closed = Event()
        self._watchdog = None
        if watchdog:
            self._watchdog = Thread(target=self._watch, daemon=True, name="handshake-watchdog")
            self._watchdog.start()

    def enable(self):
        with self.controller.gate:
            self.dispatcher.enable(self.clock())
            self.engine._sync()

    def stop(self):
        self.dispatcher.stop(self.clock())

    def update(self, evidence):
        with self.controller.gate:
            events = self.engine.update(evidence, now_ns=self.clock())
            tickets = []
            for event in events:
                command = self.mapper.map(event, self.controller.snapshot(), now_ns=self.clock())
                if command is not None:
                    tickets.append(self.dispatcher.submit(command))
            return events, tickets

    def tick(self):
        self.dispatcher.tick()
        with self.controller.gate:
            self.engine.tick(self.clock())

    def change_mode(self, mode):
        self.stop()
        with self.controller.gate:
            self.controller.change_mode(mode, self.clock())
            self.engine._sync()

    def change_model(self, model_version):
        self.stop()
        with self.controller.gate:
            self.controller.change_model(self.clock())
            self.engine.model_version = model_version
            self.engine._sync()

    def health(self, reason):
        # Do not queue health failure behind inference or UI work.
        self.dispatcher.stop(self.clock(), suspend=True)
        with self.controller.gate:
            self.engine._sync()
            self.engine._record(reason)

    def _watch(self):
        while not self._closed.wait(0.02):
            try:
                self.tick()
            except Exception:
                self.dispatcher.stop()
                return

    def close(self):
        self._closed.set()
        self.dispatcher.close()
        if self._watchdog is not None:
            self._watchdog.join(timeout=0.1)
