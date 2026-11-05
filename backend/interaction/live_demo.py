"""Explicit, experimental LIVE control of an application-local slide deck only.

This is not the shadow session, not an external PPT/keyboard backend and not a
claim that the draft event policy is production-validated. Existing shadow and
synthetic modes keep their original authorization rules.
"""

from hashlib import sha256
import json

from algorithm.bundle import digest, json_bytes

from backend.interaction.model_host import ModelHost
from backend.interaction.model_session import ModelPresentationSession
from backend.interaction.public_policy import PublicGestureEngine
from backend.interaction.public_presentation import SyntheticPublicPresentation
from shared.presentation_protocol import Deck


def local_demo_deck():
    deck = Deck.model_validate_json(
        json.dumps(
            {
                "format_version": "presentation-deck-v1",
                "title": "HandShake - experimental gesture presentation",
                "slides": [
                    {
                        "title": "HandShake",
                        "lines": [
                            "Real model -> gesture policy -> local slide page",
                            "Experimental demo, not PowerPoint or OS keyboard control.",
                        ],
                    },
                    {
                        "title": "The model",
                        "lines": [
                            "21 hand landmarks -> 94 features -> 32-frame causal TCN",
                            "15 gesture classes. Recognition is not control intent.",
                        ],
                    },
                    {
                        "title": "Explicit control",
                        "lines": [
                            "Enable, choose a neutral owner, warm up, then gesture.",
                            "TLT: previous / TRT: next. Return to neutral to release.",
                        ],
                    },
                    {
                        "title": "Stop and limitations",
                        "lines": [
                            "SPACE: stop control. ESC: close camera and exit.",
                            "This is an unverified live demo; errors remain possible.",
                        ],
                    },
                ],
            }
        )
    )
    return deck, sha256(deck.model_dump_json().encode()).hexdigest()


class _LocalDemoGestureEngine(PublicGestureEngine):
    # Selecting a stationary pointing hand must not require the classifier to
    # call that same pose NOG. These static classes still cannot create actions.
    # All confidence, quality, freshness, geometry and release-hold checks stay.
    neutral_tokens = ("NOG", "P1F", "P2F")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bind the changed local-only semantics, not the old NOG-only policy SHA.
        self.policy_sha256 = digest(json_bytes({
            "policy": self.policy.model_dump(mode="json"),
            "neutral_tokens": list(self.neutral_tokens),
            "scope": "application-local-demo",
        }))


class _LiveDemoHost(ModelHost):
    def __init__(self, *, deck, **kwargs):
        self._local_deck = deck
        super().__init__(**kwargs)

    def _make_engine(self, controller, **kwargs):
        return _LocalDemoGestureEngine(controller, **kwargs)

    def _deck(self):
        return self._local_deck

    def _validate_calibration_source(self, calibration, allow_synthetic):
        if (
            self.execution != "application_local_demo"
            or self.timebase != "monotonic"
            or self.source_synthetic is not False
            or allow_synthetic is not False
            or calibration.synthetic
            or calibration.source.data_synthetic
            or calibration.source.model_synthetic
            or calibration.rejection_status != "selected"
        ):
            raise ValueError(
                "live local demo needs real model/calibration, monotonic source and feasible rejection"
            )

    def _arm_backend(self, context, track_id):
        SyntheticPublicPresentation._arm_backend(self, context, track_id)

    def _authorized(self, context):
        return (
            self.execution == "application_local_demo"
            and self.timebase == "monotonic"
            and not self.source_synthetic
            and not self.engine.calibration.synthetic
            and self.warmup is None
            and SyntheticPublicPresentation._authorized(self, context)
        )


class LocalDemoSession(ModelPresentationSession):
    """Separate constructor/opt-in; real observations still go through the full model.

    No class overrides, input injection, arbitrary backend or dispatcher callbacks.
    Session creation does not enable control. Enable and owner selection must each
    be requested explicitly against a fresh rendered surface/epoch.
    """

    def __init__(self, *, allow_application_local=False, deck=None, **kwargs):
        if allow_application_local is not True:
            raise ValueError(
                "experimental application-local execution requires explicit opt-in"
            )
        forbidden = {
            "execution",
            "source_synthetic",
            "timebase",
            "allow_synthetic",
        } & kwargs.keys()
        if forbidden:
            raise ValueError("local demo source/mode cannot be overridden")
        self._local_deck = local_demo_deck() if deck is None else deck
        value, identity = self._local_deck
        if (
            not isinstance(value, Deck)
            or not isinstance(identity, str)
            or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
        ):
            raise ValueError("bounded local deck and SHA256 identity required")
        super().__init__(
            execution="application_local_demo",
            source_synthetic=False,
            timebase="monotonic",
            allow_synthetic=False,
            **kwargs,
        )

    def _make_host(self, **kwargs):
        return _LiveDemoHost(deck=self._local_deck, **kwargs)

    def snapshot(self):
        with self.controller.gate:
            value = super().snapshot()
            return value | {
                "format_version": "local-gesture-demo-v1",
                "live_available": True,
                "external_control_available": False,
                "experimental": True,
                "actions_enabled": bool(
                    self.host.engine.state in {"ACTIVE", "CANDIDATE"}
                    and self.host._authorized(self.controller.snapshot())
                ),
            }
