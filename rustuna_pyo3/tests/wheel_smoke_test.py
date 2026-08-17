import os
import sys
import sysconfig
import tempfile
from pathlib import Path

import rustuna


def main() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        database_path = Path(workdir) / "rustuna.db"
        storage = rustuna.storages.SQLite3Storage(
            str(database_path), create_database=True
        )
        study = rustuna.create_study(storage=storage)
        trial = study.ask()
        value = trial.suggest_float("x", 0.0, 1.0)
        study.tell(trial.number, value)

        assert database_path.is_file()
        assert len(study.trials) == 1
        assert study.best_trial.value == value

    if os.environ.get("RUSTUNA_EXPECT_FREE_THREADED") == "1":
        assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
        is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
        assert is_gil_enabled is not None
        assert not is_gil_enabled()


if __name__ == "__main__":
    main()
