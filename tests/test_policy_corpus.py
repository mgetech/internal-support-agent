"""Checks on what is derived from the hand-written policies: the version bump rule, the
chunk snapshot, and the chunk ids that other files refer to. CI runs these on every push.
"""

from pathlib import Path

import pytest
import yaml
from data.generate import PolicyFile, generate_policy_chunks, read_policies
from data.snapshot import (
    CHECKSUMS_FILE,
    CHUNK_SNAPSHOT_FILE,
    calculate_chunk_snapshot,
    calculate_policy_checksums,
    get_snapshot_problems,
    get_version_problems,
    read_json,
)
from eval.run_eval import CHUNK_ID_RE

RETRIEVAL_GROUND_TRUTH_FILE = (
    Path(__file__).resolve().parent.parent / "eval" / "retrieval_ground_truth.yaml"
)


def _policies(text="Staff get 30 days.", version="2026-09.1") -> dict[str, PolicyFile]:
    policy = PolicyFile(id="alpha", domain="hr", policy_version=version, text=text)
    return {"alpha": policy}


# --- the version bump rule ---


def test_unchanged_policy_has_no_problems():
    assert get_version_problems(_policies(), calculate_policy_checksums(_policies())) == []


def test_text_edit_without_version_bump_fails():
    recorded = calculate_policy_checksums(_policies())

    problems = get_version_problems(_policies(text="Staff get 28 days."), recorded)

    assert len(problems) == 1
    assert "the text changed but policy_version is still 2026-09.1" in problems[0]
    assert "Bump policy_version" in problems[0]


def test_text_edit_with_bump_asks_to_update_checksums():
    recorded = calculate_policy_checksums(_policies())

    problems = get_version_problems(
        _policies(text="Staff get 28 days.", version="2026-10.1"), recorded
    )

    assert len(problems) == 1
    assert "checksums.json is out of date" in problems[0]


def test_version_bump_alone_asks_to_update_checksums():
    recorded = calculate_policy_checksums(_policies())

    problems = get_version_problems(_policies(version="2026-10.1"), recorded)

    assert len(problems) == 1
    assert "checksums.json is out of date" in problems[0]


def test_new_policy_needs_a_recorded_checksum():
    problems = get_version_problems(_policies(), {})

    assert len(problems) == 1
    assert "no recorded checksum" in problems[0]


def test_removed_policy_is_reported():
    problems = get_version_problems({}, calculate_policy_checksums(_policies()))

    assert len(problems) == 1
    assert "the policy is gone" in problems[0]


def test_committed_policies_match_committed_checksums():
    recorded = read_json(CHECKSUMS_FILE)

    assert get_version_problems(read_policies(), recorded) == []


# --- the chunk snapshot ---


def test_unchanged_chunks_have_no_problems():
    snapshot = {"alpha#a#0": "1", "alpha#b#0": "2"}

    assert get_snapshot_problems(snapshot, dict(snapshot)) == []


def test_a_changed_chunk_text_is_reported():
    problems = get_snapshot_problems(
        {"alpha#a#0": "1", "alpha#b#0": "3"}, {"alpha#a#0": "1", "alpha#b#0": "2"}
    )

    assert problems[0] == "alpha#b#0: the chunk text changed"
    assert "python -m data.snapshot" in problems[-1]


def test_a_renamed_heading_is_one_removed_and_one_new_chunk():
    problems = get_snapshot_problems({"alpha#new#0": "1"}, {"alpha#old#0": "1"})

    assert "alpha#new#0: new chunk, not in chunk_snapshot.json" in problems
    assert "alpha#old#0: chunk is gone, a citation of this id would break" in problems


def test_the_chunks_match_the_committed_snapshot():
    current = calculate_chunk_snapshot(generate_policy_chunks())

    assert get_snapshot_problems(current, read_json(CHUNK_SNAPSHOT_FILE)) == []


# --- references to chunk ids resolve ---


def _chunk_ids_in(value) -> set[str]:
    """Every string in a loaded YAML value that looks like a chunk id."""
    if isinstance(value, str):
        return {value} if CHUNK_ID_RE.match(value) else set()
    children = (
        value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    )
    return {chunk_id for child in children for chunk_id in _chunk_ids_in(child)}


def test_chunk_ids_are_found_anywhere_in_a_yaml_value():
    value = {"q1": {"relevant": ["vacation-policy#entitlement#0", "not an id"]}, "n": 3}

    assert _chunk_ids_in(value) == {"vacation-policy#entitlement#0"}


def test_every_id_in_the_retrieval_ground_truth_is_a_current_chunk():
    if not RETRIEVAL_GROUND_TRUTH_FILE.exists():
        pytest.skip("eval/retrieval_ground_truth.yaml does not exist yet")
    text = RETRIEVAL_GROUND_TRUTH_FILE.read_text(encoding="utf-8")
    referenced = _chunk_ids_in(yaml.safe_load(text))
    current = {chunk["id"] for chunk in generate_policy_chunks()}

    assert referenced, "the ground truth file names no chunk id"
    assert referenced <= current, sorted(referenced - current)
