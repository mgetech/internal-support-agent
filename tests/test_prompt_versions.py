import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from support_agent import prompts
from support_agent.prompts import (
    CHECKSUMS_FILE,
    Prompt,
    checksums,
    get_prompt,
    load_prompts,
    prompt_versions,
    version_problems,
)


def _write(directory: Path, name: str, **overrides: str) -> None:
    fields = {
        "id": name,
        "version": "1.0.0",
        "text": "Answer the question.",
        "changelog": "first version",
    }
    fields.update(overrides)
    body = "\n".join(f"{key}: {value!r}" for key, value in fields.items())
    (directory / f"{name}.yaml").write_text(body, encoding="utf-8")


def test_loads_every_file_by_id(tmp_path):
    _write(tmp_path, "alpha")
    _write(tmp_path, "beta", version="2.1.0")

    loaded = load_prompts(tmp_path)

    assert set(loaded) == {"alpha", "beta"}
    assert loaded["beta"].version == "2.1.0"
    assert loaded["alpha"].text == "Answer the question."


def test_tag_holds_id_and_version(tmp_path):
    _write(tmp_path, "alpha", version="1.2.3")

    tag = load_prompts(tmp_path)["alpha"].tag

    assert tag == {"prompt_id": "alpha", "prompt_version": "1.2.3"}


def test_id_must_match_file_name(tmp_path):
    _write(tmp_path, "alpha", id="other")

    with pytest.raises(ValueError, match="must match the file name"):
        load_prompts(tmp_path)


@pytest.mark.parametrize("version", ["1", "1.0", "1.0.0.0", "v1.0.0", "1.0.x", ""])
def test_version_must_be_semver(version):
    with pytest.raises(ValidationError):
        Prompt(id="alpha", version=version, text="text", changelog="change")


@pytest.mark.parametrize("field", ["text", "changelog"])
def test_blank_fields_are_rejected(field):
    values = {"id": "alpha", "version": "1.0.0", "text": "text", "changelog": "change"}
    values[field] = "  "

    with pytest.raises(ValidationError):
        Prompt(**values)


def test_missing_field_is_rejected(tmp_path):
    (tmp_path / "alpha.yaml").write_text("id: alpha\nversion: 1.0.0\ntext: hi\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_prompts(tmp_path)


def test_unknown_id_names_the_known_ids(monkeypatch, tmp_path):
    _write(tmp_path, "alpha")
    monkeypatch.setattr(prompts, "_registry", lambda: load_prompts(tmp_path))

    with pytest.raises(KeyError, match="known: alpha"):
        get_prompt("missing")


def test_prompt_versions_lists_every_prompt(monkeypatch, tmp_path):
    _write(tmp_path, "alpha")
    _write(tmp_path, "beta", version="0.3.0")
    monkeypatch.setattr(prompts, "_registry", lambda: load_prompts(tmp_path))

    assert prompt_versions() == {"alpha": "1.0.0", "beta": "0.3.0"}


def _prompt(text="Answer the question.", version="1.0.0") -> dict[str, Prompt]:
    return {"alpha": Prompt(id="alpha", version=version, text=text, changelog="change")}


def test_unchanged_prompt_has_no_problems():
    assert version_problems(_prompt(), checksums(_prompt())) == []


def test_text_edit_without_version_bump_fails():
    recorded = checksums(_prompt())

    problems = version_problems(_prompt(text="Answer briefly."), recorded)

    assert len(problems) == 1
    assert "the text changed but the version is still 1.0.0" in problems[0]
    assert "Bump the version" in problems[0]


def test_text_edit_with_bump_asks_to_update_checksums():
    recorded = checksums(_prompt())

    problems = version_problems(_prompt(text="Answer briefly.", version="1.0.1"), recorded)

    assert len(problems) == 1
    assert "checksums.json is out of date" in problems[0]


def test_new_prompt_needs_a_recorded_checksum():
    problems = version_problems(_prompt(), {})

    assert len(problems) == 1
    assert "no recorded checksum" in problems[0]


def test_removed_prompt_is_reported():
    problems = version_problems({}, checksums(_prompt()))

    assert len(problems) == 1
    assert "the prompt is gone" in problems[0]


def test_committed_prompts_match_committed_checksums():
    recorded = json.loads(CHECKSUMS_FILE.read_text(encoding="utf-8"))

    assert version_problems(load_prompts(), recorded) == []


def test_agent_prompt_has_the_seven_rules_in_order():
    text = load_prompts()["agent_system"].text
    rule_starts = [text.find(f"\n{n}. ") for n in range(1, 8)]

    assert -1 not in rule_starts
    assert rule_starts == sorted(rule_starts)
    assert "\n8. " not in text
