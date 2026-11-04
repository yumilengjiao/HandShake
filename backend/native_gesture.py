"""Native Flutter adapter for the existing live, application-local gesture session.

The UI cannot submit logits, classes or navigation. Only DemoCameraWorker and
LocalDemoSession can change pages. Parent/surface leases remain independent of
model freshness. Camera and control both start disabled.
"""

import base64
import json
import threading
import time
from collections import OrderedDict
from hashlib import sha256
from uuid import uuid4

from shared.presentation_protocol import Deck
from shared.preview_protocol import MAX_JPEG_BYTES

SURFACE_TTL_NS = 500_000_000
# A display frame is not action evidence. Keep the preview until the input-stall
# deadline; model/owner/dispatch evidence retains its independent 150ms checks.
PREVIEW_MAX_AGE_NS = 750_000_000


def native_demo_deck():
    deck = Deck.model_validate_json(
        json.dumps(
            {
                "format_version": "presentation-deck-v1",
                "title": "HandShake · 实时手势翻页预览",
                "slides": [
                    {
                        "title": "用手势翻动这一页",
                        "lines": [
                            "打开相机 → 启用手势 → 选择控制手 → 保持静止完成预热。",
                            "向右抛动：下一页；向左抛动：上一页。每次动作后回到中立。",
                            "这里展示的是本应用的实际页码，不是识别候选或录像。",
                        ],
                    },
                    {
                        "title": "从画面到手势",
                        "lines": [
                            "MediaPipe提取21个手部关键点，构成每帧94维特征。",
                            "因果TCN读取最近32帧，输出15类手势分数。",
                            "识别结果还需满足有效性、时效、位移和释放条件才会翻页。",
                        ],
                    },
                    {
                        "title": "控制只在本窗口内",
                        "lines": [
                            "控制手必须由你明确选择，不会自动把路人的手作为控制手。",
                            "空格停止手势控制；Esc关闭相机。切换页面、失活也会停止。",
                            "不注入系统按键，不控制PowerPoint、WPS或其他应用。",
                        ],
                    },
                    {
                        "title": "完成一次真实预览",
                        "lines": [
                            "当前仍是实验性功能，可能漏识别或误识别。",
                            "离线验证分数不代表现场成功率，实时稳定性仍需实际测试。",
                            "向左抛动可回到上一页；遇到异常请先停止控制。",
                        ],
                    },
                ],
            }
        )
    )
    return deck, sha256(deck.model_dump_json().encode()).hexdigest()


class NativeGestureSession:
    def __init__(
        self,
        *,
        assets=None,
        perception_model=None,
        camera=0,
        worker_factory=None,
        clock=time.monotonic_ns,
    ):
        self.assets, self.perception_model, self.camera = (
            assets,
            perception_model,
            camera,
        )
        self.worker_factory, self.clock = worker_factory, clock
        self.lock = threading.RLock()
        self.session_id = uuid4().hex
        self.sequence = self.camera_epoch = self.last_command_id = 0
        self.last_heartbeat = clock()
        self.started = self.release_started = None
        self.worker = self.frame = None
        self.closed = False
        self.reason = "initial" if self.available else "model_not_configured"
        self.deck = native_demo_deck()
        self.offers = OrderedDict()

    @property
    def available(self):
        return bool(self.assets and self.perception_model)

    @property
    def released(self):
        return self.worker is None or (
            self.worker.done.is_set() and not self.worker.thread.is_alive()
        )

    def _adapter(self):
        return self.worker.adapter if self.worker is not None else None

    def _stop_control(self, reason):
        adapter = self._adapter()
        if adapter is not None:
            adapter.stop()
        self.offers.clear()
        self.reason = reason

    def _close_camera(self, reason):
        self._stop_control(reason)
        self.frame = None
        if self.worker is not None:
            if self.release_started is None:
                self.release_started = self.clock()
                self.camera_epoch += 1
            self.worker.request_stop()

    def fail(self, reason):
        with self.lock:
            self._close_camera(reason)
            self.closed = True

    def _expire(self):
        now = self.clock()
        if (
            now - self.last_heartbeat >= 1_500_000_000
            and self.worker is not None
            and not self.worker.stop_requested.is_set()
        ):
            self._close_camera("heartbeat_timeout")
        if now - self.last_heartbeat >= 5_000_000_000:
            self.closed = True
        if (
            self.release_started is not None
            and not self.released
            and now - self.release_started >= 2_000_000_000
        ):
            self.reason = "release_unconfirmed"
            self.closed = True

    def _offer(self, command):
        offer = self.offers.get(int(command.based_on_sequence))
        if (
            offer is None
            or int(command.camera_epoch) != self.camera_epoch
            or offer[1] != self.camera_epoch
            or not 0 <= self.clock() - offer[0] < SURFACE_TTL_NS
        ):
            return None
        return offer

    def accept(self, command):
        with self.lock:
            if self.closed:
                return
            if command.session_id != self.session_id:
                raise ValueError("wrong session")
            command_id = int(command.command_id)
            if command_id <= self.last_command_id:
                raise ValueError("non-increasing command id")
            self.last_command_id = command_id
            self._expire()
            if self.closed:
                return
            kind = command.kind
            if kind == "heartbeat":
                self.last_heartbeat = self.clock()
                return
            if kind in {"stop", "close_camera", "shutdown"}:
                if kind == "stop":
                    self._stop_control("user_stop")
                else:
                    self._close_camera(
                        "camera_closed" if kind == "close_camera" else "shutdown"
                    )
                    self.closed = kind == "shutdown"
                return
            offer = self._offer(command)
            if offer is None:
                self.reason = "stale_surface"
                return
            if kind == "open_camera":
                if not self.available:
                    self.reason = "model_not_configured"
                    return
                if not self.released:
                    self.reason = "camera_already_running"
                    return
                if self.worker_factory is None:
                    from backend.perception.local_demo import DemoCameraWorker

                    factory = DemoCameraWorker
                else:
                    factory = self.worker_factory
                self.camera_epoch += 1
                self.frame = None
                self.offers.clear()
                self.release_started = None
                self.started = self.last_heartbeat = self.clock()
                try:
                    self.worker = factory(
                        allow_application_local=True,
                        deck=self.deck,
                        device=self.camera,
                        model_path=self.perception_model,
                        model_package=self.assets["package"],
                        model_sha256=self.assets["sha256"],
                        calibration_path=self.assets["calibration"],
                        calibration_sha256=self.assets["calibration_sha256"],
                    )
                    self.worker.start()
                    self.reason = "camera_starting"
                except Exception:  # noqa: BLE001 - fail closed at native worker boundary
                    self._close_camera("camera_start_failed")
                    self.closed = True
                return
            adapter, snapshot = self._adapter(), offer[2]
            if (
                adapter is None
                or snapshot is None
                or self.worker.stop_requested.is_set()
                or self.worker.done.is_set()
                or adapter.session is None
            ):
                self.reason = "camera_not_ready"
                return
            # Check the original host token, not a freshly created replacement.
            host = adapter.session.host
            with adapter.session.controller.gate:
                if not host._receipt(
                    snapshot["sequence"], snapshot["epoch"], snapshot["revision"]
                ):
                    self.reason = "stale_surface"
                    return
                if kind == "surface_ack":
                    adapter.acknowledge(snapshot)
                elif kind == "enable":
                    self.reason = (
                        "enabled" if adapter.enable(snapshot) else "enable_rejected"
                    )
                elif kind == "choose_owner":
                    self.reason = (
                        "owner_requested"
                        if adapter.request_owner(snapshot)
                        else "owner_rejected"
                    )

    def snapshot(self):
        with self.lock:
            self._expire()
            now = self.clock()
            worker = self.worker
            if worker is not None and not worker.stop_requested.is_set():
                packet = worker.take()
                if packet is not None:
                    self.frame = packet
                if worker.error or worker.done.is_set():
                    self._close_camera(worker.error or "camera_unavailable")
                elif (
                    self.frame is None
                    and now - self.started >= 10_000_000_000
                    or self.frame is not None
                    and now - self.frame.timestamp_ns > PREVIEW_MAX_AGE_NS
                ):
                    self._close_camera("input_stalled")
            camera_state = "stopped"
            if worker is not None and not self.released:
                camera_state = (
                    "stopping"
                    if worker.stop_requested.is_set()
                    else ("running" if self.frame is not None else "starting")
                )
            adapter = self._adapter()
            demo = adapter.snapshot() if adapter is not None else None
            page = demo["page"] if demo else 1
            slide = self.deck[0].slides[page - 1]
            enabled = owner = False
            if adapter is not None and adapter.session is not None:
                context = adapter.session.controller.snapshot()
                enabled = context.enabled and not context.suspended
                owner = context.owner_track_id is not None
            active = camera_state == "running"
            frame = self.frame if active else None
            if frame is not None and not (
                0 <= now - frame.timestamp_ns <= PREVIEW_MAX_AGE_NS
            ):
                frame = None
            jpeg = frame.jpeg if frame else b""
            if len(jpeg) > MAX_JPEG_BYTES:
                self.fail("image_budget_exceeded")
                jpeg = b""
            classification = demo.get("classification") if demo else None
            top_class = None
            if classification and classification["state"] == "predicted":
                logits = classification["logits"]
                top_class = adapter.bundle.vocabulary.tokens[
                    max(range(len(logits)), key=logits.__getitem__)
                ]
            self.sequence += 1
            self.offers[self.sequence] = (now, self.camera_epoch, demo)
            while len(self.offers) > 32:
                self.offers.popitem(last=False)
            safety = demo.get("safety", {}) if demo else {}
            return {
                "protocol": "native-gesture-v1",
                "version": 1,
                "session_id": self.session_id,
                "sequence": str(self.sequence),
                "camera_epoch": str(self.camera_epoch),
                "last_command_id": str(self.last_command_id),
                "timestamp_ns": str(now),
                "scope": "application-local",
                "external_control_available": False,
                "experimental": True,
                "model_available": self.available,
                "camera_state": camera_state,
                "camera_released": self.released,
                "reason": self.reason,
                "enabled": bool(active and enabled),
                "owner_bound": bool(active and owner),
                "eligible_owner": bool(
                    active and enabled and demo and demo.get("eligible_owner")
                ),
                "owner_pending": bool(demo and demo.get("owner_request_pending")),
                "warming": bool(demo and demo.get("model_warming")),
                "busy": bool(demo and (demo.get("model_busy") or demo.get("busy"))),
                "actions_enabled": bool(
                    active and enabled and demo and demo.get("actions_enabled")
                ),
                "control_released": bool(
                    not enabled and safety.get("safe_quiescent", True)
                ),
                "model_state": demo["state"] if demo else "DISABLED",
                "policy_reason": demo["bridge_reason"] if demo else "waiting_camera",
                "top_class": top_class,
                "model_version": demo["model_version"] if demo else None,
                "model_epoch": str(demo["epoch"]) if demo else None,
                "revision": str(demo["revision"]) if demo else "0",
                "rendered": bool(demo and demo["rendered"]),
                "page": page,
                "page_count": len(self.deck[0].slides),
                "title": slide.title,
                "lines": list(slide.lines),
                "jpeg": base64.b64encode(jpeg).decode("ascii"),
            }

    def close(self):
        with self.lock:
            self._close_camera("shutdown")
            self.closed = True
        if self.worker is not None:
            self.worker.thread.join(timeout=0.3)
