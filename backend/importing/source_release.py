"""A reviewed, immutable source snapshot; never accept arbitrary mismatched downloads.

Publisher checksums remain in the catalog. Only an explicit project release can
select a different annotation archive, with its discrepancy and audit preserved.
"""

from hashlib import sha256
from importlib.resources import files

from algorithm.data.source_archive import VerifiedArchive, strict_json

RELEASE_ID = "ipn-hands-frames-v1"
RESOURCE = "ipn_hands_release.json"
CAPTURE_GROUPING = {
    "unit": "source_capture_group",
    "subject_identity_verified": False,
    "legacy_subject_id_field": "capture-group namespace, NOT verified person identifiers",
}


def load_release(reference):
    raw = files("shared").joinpath("defaults", RESOURCE).read_bytes()
    if (
        not isinstance(reference, dict)
        or set(reference) != {"id", "sha256"}
        or reference["id"] != RELEASE_ID
        or reference["sha256"] != sha256(raw).hexdigest()
    ):
        raise ValueError("project source review reference/hash mismatch")
    value = strict_json(raw)
    if (
        value.get("format_version") != "ipn-hands-project-release-v1"
        or value.get("release_id") != RELEASE_ID
        or value.get("status") != "project_reviewed"
    ):
        raise ValueError("unapproved project source release")
    return value


def resolve_source_entries(source):
    items = source["files"]
    result = {entry["filename"]: dict(entry) for entry in items}
    if len(result) != len(items):
        raise ValueError("duplicate source archive filename")
    reference = source.get("project_release")
    if reference is None:
        return result
    release = load_release(reference)
    if source.get("upstream_commit") != release["upstream_commit"]:
        raise ValueError("project review belongs to another upstream version")
    for name, pin in release["archives"].items():
        entry = result[name]
        if entry["expected_sha256"] != pin["published_sha256"]:
            raise ValueError(
                "published checksum must remain recorded, not silently replaced"
            )
        entry["published_sha256"] = entry["expected_sha256"]
        entry["expected_sha256"] = pin["accepted_sha256"]
        entry["project_release"] = reference
    return result


def verify_release_annotations(path, release):
    name = "IPN_HandS_annotations.zip"
    pin = release["archives"][name]
    with VerifiedArchive(
        path,
        expected_sha256=pin["accepted_sha256"],
        max_member_bytes=2 * 1024**2,
        max_total_bytes=32 * 1024**2,
    ) as archive:
        if set(archive.members) != set(pin["members"]):
            raise ValueError("reviewed annotation member inventory mismatch")
        for member, expected in pin["members"].items():
            raw = archive.read_bytes(member)
            if (
                len(raw) != expected["bytes"]
                or sha256(raw).hexdigest() != expected["sha256"]
            ):
                raise ValueError("reviewed annotation content changed")


def capture_plan(videos, release):
    """Fixed capture groups, NOT guessed participant identities or random frame splits."""
    inventory = release["sequences"]
    if set(videos) != set(inventory):
        raise ValueError("source sequence inventory differs from reviewed release")
    groups = {}
    for name, source in videos.items():
        item = inventory[name]
        if (
            source.frame_count != item["frames"]
            or source.source_split != item["source_split"]
        ):
            raise ValueError("reviewed sequence timing/split mismatch")
        group = item["capture_group"]
        if group in groups and groups[group] != source.source_split:
            raise ValueError("capture group leaks into official test")
        groups[group] = source.source_split
    candidates = sorted(
        (g for g, role in groups.items() if role == "train"),
        key=lambda g: sha256((release["release_id"] + "/" + g).encode()).hexdigest(),
    )
    count = release["grouping"]["validation_groups"]
    if type(count) is not int or not 0 < count < len(candidates):
        raise ValueError("invalid fixed validation group budget")
    for group in candidates[:count]:
        groups[group] = "val"
    return {
        "format_version": "capture-group-plan-v1",
        "status": "reviewed",
        "reference": "project-release:" + release["release_id"],
        # Legacy field names are retained for cache compatibility. Their meaning
        # is explicit in every import/cache/model's grouping declaration.
        "subject_splits": dict(sorted(groups.items())),
        "videos": {
            name: {"subject_id": item["capture_group"], "session_id": name}
            for name, item in sorted(inventory.items())
        },
    }


def plan_grouping(plan):
    if plan.get("format_version") == "capture-group-plan-v1":
        return dict(CAPTURE_GROUPING)
    return None  # legacy explicitly reviewed subject plans retain their contract
