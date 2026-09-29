"""Versioned prompt registry. Each file in `prompts/` holds one prompt: id, semver
version, text and a one-line changelog. Prompts are loaded once and read by id.

Every caller that sends a prompt to a model adds `prompt.tag` to its trace, audit
payload and Decision Record, so a result can be traced back to the exact version.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
CHECKSUMS_FILE = PROMPTS_DIR / "checksums.json"


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    version: str
    text: str
    changelog: str

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        parts = value.split(".")
        if len(parts) != 3 or not all(p.isascii() and p.isdigit() for p in parts):
            raise ValueError(f"version must look like 1.0.0, got {value!r}")
        return value

    @field_validator("id", "text", "changelog")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @property
    def tag(self) -> dict[str, str]:
        """The id and version to record wherever this prompt is used."""
        return {"prompt_id": self.id, "prompt_version": self.version}


def load_prompts(directory: Path = PROMPTS_DIR) -> dict[str, Prompt]:
    """Read every `*.yaml` file in `directory`. The id must match the file name,
    so an id can only appear once.
    """
    prompts: dict[str, Prompt] = {}
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}: expected a mapping at the top level")
        prompt = Prompt(**data)
        if prompt.id != path.stem:
            raise ValueError(f"{path.name}: id {prompt.id!r} must match the file name")
        prompts[prompt.id] = prompt
    return prompts


@lru_cache(maxsize=1)
def _registry() -> dict[str, Prompt]:
    return load_prompts()


def get_prompt(prompt_id: str) -> Prompt:
    try:
        return _registry()[prompt_id]
    except KeyError:
        known = ", ".join(sorted(_registry())) or "none"
        raise KeyError(f"unknown prompt id {prompt_id!r} (known: {known})") from None


def prompt_versions() -> dict[str, str]:
    """Every loaded prompt id with its version. Eval runs store this as the
    prompt-version set they ran against.
    """
    return {prompt_id: prompt.version for prompt_id, prompt in sorted(_registry().items())}


def checksums(loaded: dict[str, Prompt]) -> dict[str, dict[str, str]]:
    """The version and a SHA-256 hash of the text for each prompt."""
    result = {}
    for prompt_id, prompt in sorted(loaded.items()):
        result[prompt_id] = {
            "version": prompt.version,
            "sha256": hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(),
        }
    return result


def version_problems(loaded: dict[str, Prompt], recorded: dict[str, dict[str, str]]) -> list[str]:
    """Compare the loaded prompts with the committed checksums. Returns one message
    for each problem. An empty list means every text change has a version bump.
    """
    update = "run `python -m support_agent.prompts` to update prompts/checksums.json"
    problems = []
    for prompt_id, current in checksums(loaded).items():
        old = recorded.get(prompt_id)
        if old is None:
            problems.append(f"{prompt_id}: no recorded checksum, {update}")
        elif current["sha256"] != old["sha256"] and current["version"] == old["version"]:
            problems.append(
                f"{prompt_id}: the text changed but the version is still {old['version']}. "
                f"Bump the version, then {update}"
            )
        elif current != old:
            problems.append(f"{prompt_id}: checksums.json is out of date, {update}")
    problems.extend(
        f"{prompt_id}: recorded in checksums.json but the prompt is gone, {update}"
        for prompt_id in sorted(set(recorded) - set(loaded))
    )
    return problems


if __name__ == "__main__":
    CHECKSUMS_FILE.write_text(json.dumps(checksums(load_prompts()), indent=2) + "\n")
