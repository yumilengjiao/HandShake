import json

import pytest

from shared.classes import ClassVocabulary, load_classes


def test_packaged_vocabulary_is_single_source():
    vocabulary = load_classes()
    assert vocabulary.class_count == 5
    assert vocabulary.id_for("swipe_left") == 2
    assert len(vocabulary.digest()) == 64
    with pytest.raises(ValueError):
        vocabulary.id_for("unknown")


@pytest.mark.parametrize("kind", ["order", "duplicate_id", "float_id", "count", "version", "extra", "bool_count"])
def test_changed_or_malformed_vocabulary_fails(tmp_path, kind):
    data = load_classes().to_dict()
    if kind == "order":
        data["classes"][2]["token"] = "swipe_right"
    elif kind == "duplicate_id":
        data["classes"][2]["id"] = 1
    elif kind == "float_id":
        data["classes"][2]["id"] = 2.0
    elif kind == "count":
        data["class_count"] = 6
    elif kind == "version":
        data["vocabulary_version"] = "new"
    elif kind == "bool_count":
        data["class_count"] = True
    else:
        data["extra"] = 1
    path = tmp_path / "classes.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_classes(path)
