import json

import pytest

from scan_queue.generator import QueueGenerator
from scan_queue.schema import SchemaError, validate_record


REPLAY_FILE = "scan_queue/replay_data.json"


def load_records():
    with open(REPLAY_FILE, encoding="utf-8") as file:
        return json.load(file)


def test_replay_generates_queue():
    generator = QueueGenerator()
    queue = generator.generate_replay(REPLAY_FILE)

    assert len(queue) == 4
    assert not generator.errors


def test_empty_queue():
    generator = QueueGenerator()

    queue = generator.generate_live([])

    assert queue == []


def test_malformed_record_is_reported():
    generator = QueueGenerator()

    queue = generator.generate_live([
        {"scan_id": "INVALID"}
    ])

    assert queue == []
    assert len(generator.errors) == 1


def test_duplicate_scan_id_is_reported():
    generator = QueueGenerator()
    records = load_records()

    queue = generator.generate_live([
        records[0],
        records[0],
    ])

    assert len(queue) == 1
    assert len(generator.errors) == 1
    assert "Duplicate" in generator.errors[0]["error"]


def test_failed_recovery_cannot_be_accepted():
    record = load_records()[3]

    record["verdict"] = "accepted"

    with pytest.raises(SchemaError):
        validate_record(record)


def test_mandatory_escalation_cannot_be_accepted():
    record = load_records()[0]

    record["mandatory_escalation"] = True

    with pytest.raises(SchemaError):
        validate_record(record)


def test_missing_uncertainty_is_explicit():
    record = load_records()[0]

    record["uncertainty"] = {
        "entropy": None,
        "mc_variance": None,
        "model_disagreement": None,
    }

    validate_record(record)