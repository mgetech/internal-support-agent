"""Publishing copies one local run to the folder whose files are committed."""

import pytest
from eval import publish
from eval.publish import main, publish_run

RUN = "20261105T091500Z"


@pytest.fixture
def folders(tmp_path, monkeypatch):
    runs = tmp_path / ".runs"
    runs.mkdir()
    (runs / f"{RUN}.md").write_text("# Eval results\n", encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setattr(publish, "RUNS_DIR", runs)
    monkeypatch.setattr(publish, "RESULTS_DIR", results)
    return runs, results


def test_a_run_is_copied_under_its_own_name(folders):
    runs, results = folders

    target = publish_run(RUN, runs, results)

    assert target == results / f"{RUN}.md"
    assert target.read_text(encoding="utf-8") == "# Eval results\n"
    assert (runs / f"{RUN}.md").exists()


def test_the_results_folder_is_created_when_it_is_missing(folders):
    runs, results = folders
    assert not results.exists()

    publish_run(RUN, runs, results)

    assert results.is_dir()


def test_an_unknown_run_is_refused_and_the_known_runs_are_named(folders):
    runs, results = folders

    with pytest.raises(FileNotFoundError, match=f"runs: {RUN}"):
        publish_run("20260101T000000Z", runs, results)


def test_a_published_result_is_never_overwritten(folders):
    runs, results = folders
    publish_run(RUN, runs, results)
    (runs / f"{RUN}.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_run(RUN, runs, results)

    assert (results / f"{RUN}.md").read_text(encoding="utf-8") == "# Eval results\n"


@pytest.mark.parametrize("name", ["", "..", "../secret", "a/b", "a\\b", ".hidden"])
def test_a_name_that_is_not_a_plain_file_name_is_refused(folders, name):
    runs, results = folders

    with pytest.raises(ValueError, match="is not a run name"):
        publish_run(name, runs, results)


def test_main_prints_the_new_path_and_returns_zero(folders, capsys):
    _, results = folders

    assert main([RUN]) == 0
    assert f"Published {results / (RUN + '.md')}" in capsys.readouterr().out
    assert (results / f"{RUN}.md").is_file()


def test_main_names_the_problem_and_returns_one(folders, capsys):
    assert main(["../x"]) == 1
    assert "ERROR" in capsys.readouterr().out
