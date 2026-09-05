"""Bootstrap a ready POC profile into the project Python environment."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path


def run_profile(
    profile_arguments: Sequence[str], runner_module: str = "karate_gnn_neo4j"
) -> int:
    project_environment = Path(__file__).resolve().parents[1] / ".venv"
    project_python = project_environment / "bin" / "python"
    if (
        project_python.is_file()
        and Path(sys.prefix).resolve() != project_environment.resolve()
    ):
        entry_point = str(Path(sys.argv[0]).resolve())
        os.execv(str(project_python), [str(project_python), entry_point, *sys.argv[1:]])

    cli = import_module(runner_module).cli
    return cli([*profile_arguments, *sys.argv[1:]])
