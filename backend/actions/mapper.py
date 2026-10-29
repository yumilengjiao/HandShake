"""Map physical tokens into explicitly configured mode/target commands."""
from uuid import uuid4

from shared.config import load_config
from shared.contracts import ACTION_CAPABILITIES, ActionCommand


class ActionMapper:
    def __init__(self, config=None):
        self.config = config or load_config()

    def map(self, event, context, *, now_ns):
        if (event.stream_id, event.context_epoch, event.timebase, event.mode, event.track_id) != (
            context.stream_id, context.context_epoch, context.timebase, context.mode, context.owner_track_id
        ) or not context.enabled or context.suspended:
            return None
        mapping = self.config.actions.mappings.get(context.mode)
        action = None if mapping is None else mapping[event.gesture_id]
        if action is None:
            return None
        if not event.source_timestamp_ns <= event.commit_timestamp_ns <= now_ns or now_ns - event.source_timestamp_ns > self.config.interaction.max_prediction_age_ns:
            raise ValueError("expired_or_future_event")
        # TTL begins at source event confirmation, not whenever a queued event is mapped.
        expiry = min(event.commit_timestamp_ns + self.config.runtime.command_ttl_ns,
                     event.source_timestamp_ns + self.config.interaction.max_prediction_age_ns)
        if now_ns >= expiry:
            raise ValueError("expired_event")
        return ActionCommand(command_id=str(uuid4()), source_event_id=event.event_id, stream_id=event.stream_id,
                             context_epoch=event.context_epoch, timebase=event.timebase, action=action,
                             issued_at_ns=now_ns, expires_at_ns=expiry, target_context=context.target_context,
                             required_capabilities=[ACTION_CAPABILITIES[action]])
