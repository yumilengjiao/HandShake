"""Versioned JSONL for synthetic evidence and future runtime telemetry.

Frame images and annotation truth are intentionally not accepted by this logger.
T06 NPZ observation recordings live in shared.recordings; T23 full replay is pending.
"""

from pathlib import Path
from typing import Any

from pydantic import Field, StrictBool

from shared.contracts import ActionCommand, ActionResult, ContextChange, Contract, FrameStatus, GestureEvent, HealthEvent, Identifier, ModelPrediction, Ns

LOG_TYPES = {cls.__name__: cls for cls in (FrameStatus, ModelPrediction, GestureEvent, ActionCommand, ActionResult, ContextChange, HealthEvent)}


class LogEnvelope(Contract):
    run_id: Identifier
    timestamp_ns: Ns
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_version: Identifier
    synthetic: StrictBool
    record_type: Identifier
    payload: dict[str, Any]


def make_log(record: Contract, *, run_id: str, timestamp_ns: int, config_sha256: str, code_version: str, synthetic: bool) -> LogEnvelope:
    if type(record).__name__ not in LOG_TYPES:
        raise ValueError("this logger excludes images and annotation truth")
    return LogEnvelope(
        run_id=run_id, timestamp_ns=timestamp_ns, config_sha256=config_sha256,
        code_version=code_version, synthetic=synthetic, record_type=type(record).__name__,
        payload=record.model_dump(mode="json"), timebase=record.timebase,
    )


def write_jsonl(path: str | Path, records: list[LogEnvelope]) -> None:
    # Exclusive creation keeps a prior run's evidence from being silently overwritten.
    with Path(path).open("x", encoding="utf-8") as output:
        for record in records:
            output.write(record.model_dump_json() + "\n")


def read_jsonl(path: str | Path) -> list[LogEnvelope]:
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            envelope = LogEnvelope.model_validate_json(line)
            cls = LOG_TYPES.get(envelope.record_type)
            if cls is None:
                raise ValueError(f"line {line_number}: unknown record_type")
            payload = cls.model_validate(envelope.payload)
            if payload.timebase != envelope.timebase:
                raise ValueError(f"line {line_number}: payload timebase mismatch")
            records.append(envelope)
    return records
