"""Checks on what is derived from the hand-written policies in `data/policies/`.

Two committed files record the last reviewed state:

- `checksums.json`: the version and a SHA-256 hash of the text of each policy. Editing
  a policy's text without bumping its `policy_version` fails a test, like prompts do.
- `chunk_snapshot.json`: every chunk id with a SHA-256 hash of its content. A change to
  a chunk id or to chunk text fails a test until the snapshot is updated on purpose.

After a deliberate policy edit, bump `policy_version`, then run
`python -m data.snapshot` and review the diff of both files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.generate import POLICIES_DIR, PolicyFile, generate_policy_chunks, read_policies

CHECKSUMS_FILE = POLICIES_DIR / "checksums.json"
CHUNK_SNAPSHOT_FILE = POLICIES_DIR / "chunk_snapshot.json"

UPDATE_COMMAND = "run `python -m data.snapshot` to update"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calculate_policy_checksums(policies: dict[str, PolicyFile]) -> dict[str, dict[str, str]]:
    """The version and a SHA-256 hash of the text for each policy. The frontmatter is not
    part of the hash, so a version bump alone does not change it.
    """
    return {
        policy_id: {"version": policy.policy_version, "sha256": _sha256(policy.text)}
        for policy_id, policy in sorted(policies.items())
    }


def get_version_problems(
    policies: dict[str, PolicyFile], recorded: dict[str, dict[str, str]]
) -> list[str]:
    """Compare the policies with the committed checksums. Returns one message for each
    problem. An empty list means every text change has a version bump.
    """
    update = f"{UPDATE_COMMAND} data/policies/checksums.json"
    problems = []
    for policy_id, current in calculate_policy_checksums(policies).items():
        old = recorded.get(policy_id)
        if old is None:
            problems.append(f"{policy_id}: no recorded checksum, {update}")
        elif current["sha256"] != old["sha256"] and current["version"] == old["version"]:
            problems.append(
                f"{policy_id}: the text changed but policy_version is still {old['version']}. "
                f"Bump policy_version in the file, then {update}"
            )
        elif current != old:
            problems.append(f"{policy_id}: checksums.json is out of date, {update}")
    problems.extend(
        f"{policy_id}: recorded in checksums.json but the policy is gone, {update}"
        for policy_id in sorted(set(recorded) - set(policies))
    )
    return problems


def calculate_chunk_snapshot(chunks: list[dict]) -> dict[str, str]:
    """Chunk id -> SHA-256 hash of the chunk content, for the rows `generate_policy_chunks`
    returns.
    """
    return {
        chunk["id"]: _sha256(chunk["content"]) for chunk in sorted(chunks, key=lambda c: c["id"])
    }


def get_snapshot_problems(current: dict[str, str], recorded: dict[str, str]) -> list[str]:
    """Compare the chunks with the committed snapshot. Returns one message for each added,
    removed or changed chunk. A changed id shows up as one removed and one added chunk.
    """
    problems = [
        f"{chunk_id}: new chunk, not in chunk_snapshot.json"
        for chunk_id in sorted(set(current) - set(recorded))
    ]
    problems.extend(
        f"{chunk_id}: chunk is gone, a citation of this id would break"
        for chunk_id in sorted(set(recorded) - set(current))
    )
    problems.extend(
        f"{chunk_id}: the chunk text changed"
        for chunk_id in sorted(set(current) & set(recorded))
        if current[chunk_id] != recorded[chunk_id]
    )
    if problems:
        problems.append(f"If the change is intended, {UPDATE_COMMAND} chunk_snapshot.json")
    return problems


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    write_json(CHECKSUMS_FILE, calculate_policy_checksums(read_policies()))
    write_json(CHUNK_SNAPSHOT_FILE, calculate_chunk_snapshot(generate_policy_chunks()))
