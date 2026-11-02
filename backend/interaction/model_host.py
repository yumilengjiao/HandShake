"""Internal host: bounded, non-authorizing model warmup and mandatory shadow isolation."""

from dataclasses import dataclass

from backend.interaction.geometry import frame_geometry
from backend.interaction.public_presentation import SyntheticPublicPresentation


@dataclass(frozen=True)
class Warmup:
    epoch: int
    track: str
    deadline_ns: int
    source_ns: int
    frame_id: int
    length: int = 0


class ModelHost(SyntheticPublicPresentation):
    """Only ModelPresentationSession should construct this implementation detail."""

    def __init__(
        self,
        *,
        execution,
        timebase,
        source_synthetic,
        warmup_ns,
        source_stream_id=None,
        **kwargs,
    ):
        self.source_stream_id = source_stream_id
        self.execution = execution
        self.timebase = timebase
        self.source_synthetic = source_synthetic
        self.warmup_ns = warmup_ns
        self.warmup = None
        super().__init__(**kwargs)

    def _stream_id(self):
        return (
            self.source_stream_id
            if self.source_stream_id is not None
            else super()._stream_id()
        )

    def _source_timebase(self):
        return self.timebase

    def _validate_calibration_source(self, calibration, allow_synthetic):
        if self.execution == "synthetic_application_local":
            if self.timebase != "synthetic" or self.source_synthetic is not True:
                raise ValueError("application execution is synthetic-only")
            super()._validate_calibration_source(calibration, allow_synthetic)
        elif self.execution == "shadow":
            if (
                self.source_synthetic or calibration.synthetic
            ) and allow_synthetic is not True:
                raise ValueError("synthetic input/model requires explicit opt-in")
        else:
            raise ValueError("unsupported execution mode")

    def _arm_backend(self, context, track_id):
        if self.execution == "synthetic_application_local":
            super()._arm_backend(context, track_id)
        # Shadow keeps the backend stop bit set, in addition to denying dispatch.

    def _authorized(self, context):
        return (
            self.execution == "synthetic_application_local"
            and self.warmup is None
            and super()._authorized(context)
        )

    def _suspend_dispatch(self, now_ns, *, suspend=True):
        self.warmup = None
        super()._suspend_dispatch(now_ns, suspend=suspend)

    def arm_owner(self, track_id, prediction, frame, *, ready):
        with self.controller.gate:
            if not super().arm_owner(track_id, prediction, frame, ready=ready):
                return False
            # Epoch just changed. No old samples/logits may be relabelled.
            context = self.controller.snapshot()
            self.warmup = Warmup(
                context.context_epoch,
                track_id,
                self._read_clock() + self.warmup_ns,
                prediction.source_timestamp_ns,
                prediction.source_frame_id,
            )
            self.reason = "model_rewarming_after_owner_binding"
            return True

    def _engine_tick(self, now_ns):
        warm = self.warmup
        if warm is None:
            super()._engine_tick(now_ns)
            return
        context = self.controller.snapshot()
        if (context.context_epoch, context.owner_track_id) != (warm.epoch, warm.track):
            self.stop("warmup_context_changed")
        elif now_ns >= warm.deadline_ns:
            self.stop("model_warmup_deadline")
        elif now_ns - warm.source_ns > self.engine.policy.max_age_ns:
            self.stop("model_warmup_no_progress")

    def warmup_progress(self, notice, frame):
        """Progress is a fresh completed model/window operation, not a camera heartbeat.

        It never advances gesture candidate, release or cooldown timers. A fixed
        deadline cannot be renewed by progress. Invalid/missing evidence revokes.
        """
        with self.controller.gate:
            self.tick()
            warm = self.warmup
            if warm is None:
                return False
            context = self.controller.snapshot()
            info = frame.info
            geometry = frame_geometry(frame, warm.track)
            now = self._read_clock()
            valid = (
                notice.context_epoch == str(warm.epoch) == str(context.context_epoch)
                and notice.model_version == self.engine.model_version
                and notice.stream_id == info.stream_id == context.stream_id
                and info.timebase == context.timebase
                and notice.source_frame_id == str(info.frame_id)
                and notice.source_timestamp_ns == str(info.timestamp_ns)
                and notice.track_id == warm.track
                and info.frame_id > warm.frame_id
                and 0
                < info.timestamp_ns - warm.source_ns
                <= self.engine.policy.max_gap_ns
                and 0 <= now - info.timestamp_ns <= self.engine.policy.max_age_ns
                and warm.length < notice.window_length <= 32
                and notice.state in {"warming_up", "predicted"}
                and geometry.visible_tracks == (warm.track,)
                and geometry.quality_valid
                and not geometry.ambiguous
                and info.transform.rotation_degrees == 0
            )
            if not valid:
                self.stop("invalid_model_warmup_evidence")
                return False
            if notice.state == "predicted":
                if notice.window_length != 32:
                    self.stop("incomplete_model_warmup")
                    return False
                self.warmup = None
                self.engine.last_good_ns = info.timestamp_ns
                self.reason = "model_ready_wait_fresh_neutral"
                return True
            self.warmup = Warmup(
                warm.epoch,
                warm.track,
                warm.deadline_ns,
                info.timestamp_ns,
                info.frame_id,
                notice.window_length,
            )
            self.reason = "model_rewarming_no_action_authority"
            return False

    def update(self, prediction, frame, *, ready):
        with self.controller.gate:
            if self.warmup is not None:
                self.reason = "model_rewarming_no_action_authority"
                return None, None
            return super().update(prediction, frame, ready=ready)
