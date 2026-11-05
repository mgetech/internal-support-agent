"""Copy one run from eval/.runs/ to eval/results/, the folder whose files are committed.

    python -m eval.publish 20261105T091500Z

The name is the file name of the run, without `.md`. The run keeps its name. A published
result is never overwritten, because the committed results are the history.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from eval.run_eval import RESULTS_DIR, RUNS_DIR

# a plain file name: no folder, and not hidden
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def publish_run(name: str, runs_dir: Path = RUNS_DIR, results_dir: Path = RESULTS_DIR) -> Path:
    """Copy the run `name` to the results folder and return the new path. Raises
    ValueError for a name that is not a plain file name, FileNotFoundError when there is no
    such run, and FileExistsError when it is already published.
    """
    if not NAME_RE.match(name):
        raise ValueError(f"{name!r} is not a run name like 20261105T091500Z")
    source = runs_dir / f"{name}.md"
    if not source.is_file():
        runs = ", ".join(sorted(path.stem for path in runs_dir.glob("*.md"))) or "none"
        raise FileNotFoundError(f"no run named {name} (runs: {runs})")
    target = results_dir / f"{name}.md"
    if target.exists():
        raise FileExistsError(f"{target} exists already")
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy one run to eval/results/ to commit it.")
    parser.add_argument("name", help="the run to copy, for example 20261105T091500Z")
    args = parser.parse_args(argv)
    try:
        target = publish_run(args.name, RUNS_DIR, RESULTS_DIR)
    except (ValueError, OSError) as error:
        print(f"ERROR {error}")
        return 1
    print(f"Published {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
