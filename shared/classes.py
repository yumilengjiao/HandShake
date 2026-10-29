"""Single, packaged T05 vocabulary consumed by every model and dataset."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path

from shared import PROTOCOL_VERSION
from shared.contracts import GESTURE_IDS, Gesture


@dataclass(frozen=True)
class ClassVocabulary:
    tokens: tuple[str, ...]
    vocabulary_version: str = PROTOCOL_VERSION
    schema_version: str = "classes-v1"

    def __post_init__(self):
        expected = tuple(gesture.value for gesture in GESTURE_IDS)
        if self.schema_version != "classes-v1" or self.vocabulary_version != PROTOCOL_VERSION:
            raise ValueError("unsupported class/vocabulary version")
        if type(self.tokens) is not tuple or self.tokens != expected:
            raise ValueError("v1 classes must retain all five T05 tokens in their fixed id order")

    @property
    def class_count(self) -> int:
        return len(self.tokens)

    def id_for(self, token: str | Gesture) -> int:
        value = token.value if isinstance(token, Gesture) else token
        try:
            return self.tokens.index(value)
        except ValueError as error:
            raise ValueError(f"unknown class token: {value}") from error

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "vocabulary_version": self.vocabulary_version,
            "class_count": self.class_count,
            "classes": [{"id": i, "token": token} for i, token in enumerate(self.tokens)],
        }

    def digest(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_classes(path: str | Path | None = None) -> ClassVocabulary:
    resource = Path(path) if path is not None else files("shared").joinpath("defaults/classes.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    if set(data) != {"schema_version", "vocabulary_version", "class_count", "classes"}:
        raise ValueError("invalid classes.json fields")
    entries = data["classes"]
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) or set(item) != {"id", "token"}
        or type(item["id"]) is not int or item["id"] != index
        or type(item["token"]) is not str
        for index, item in enumerate(entries)
    ):
        raise ValueError("classes.json requires sequential integer IDs and string tokens")
    vocabulary = ClassVocabulary(
        tokens=tuple(item["token"] for item in entries),
        vocabulary_version=data["vocabulary_version"], schema_version=data["schema_version"],
    )
    if type(data["class_count"]) is not int or data["class_count"] != vocabulary.class_count:
        raise ValueError("class_count does not match the class table")
    return vocabulary
