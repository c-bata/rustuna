from __future__ import annotations

import json
import os
import tempfile

import optuna
import pytest
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import rustuna


def test_journal_file_storage_can_be_used_by_native_sampler() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        storage = rustuna.storages.JournalFileStorage(f"{workdir}/test.journal")
        study = rustuna.create_study(storage=storage)
        trial = study.ask()
        ctx = rustuna.samplers.SamplerContext(
            study_id=study._study_id,
            trial_number=trial.number,
            trial_id=trial._trial_id,
            directions=study.directions,
        )

        value = rustuna.samplers.RandomSampler(seed=1).sample_independent(
            ctx,
            storage,
            "x",
            rustuna.distributions.FloatDistribution(0, 1),
        )

        assert 0 <= value <= 1


def test_reading_trials_after_late_user_attr_write_keeps_trials_readable() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/test.journal"
        storage = rustuna.storages.JournalFileStorage(file_path)
        study = rustuna.create_study(storage=storage, study_name="reproduce-journal")

        trial = study.ask()
        storage.set_trial_user_attrs(
            trial._trial_id,
            {
                "x": "1",
                "y": "2",
            },
        )
        study.tell(trial.number, values=1.0)

        with pytest.raises(rustuna.exceptions.UpdateFinishedTrialError):
            storage.set_trial_user_attrs(
                trial._trial_id,
                {
                    "x": "1",
                    "y": "2",
                },
            )

        trials = study.trials
        assert len(trials) == 1
        assert trials[0].number == trial.number


def test_create_new_trial_from_template_optuna_compatibility() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/rustuna.journal"
        rustuna_storage = rustuna.storages.JournalFileStorage(file_path)
        rustuna_study = rustuna.create_study(
            storage=rustuna_storage, study_name="example"
        )
        rustuna_study.add_trial(
            rustuna.trial.PersistedTrial(
                trial_id=0,
                study_id=0,
                number=0,
                state=rustuna.trial.TrialState.WAITING,
            )
        )

        optuna_storage = JournalStorage(JournalFileBackend(file_path))
        studies = optuna_storage.get_all_studies()
        assert len(studies) == 1
        trials = optuna_storage.get_all_trials(studies[0]._study_id, deepcopy=False)
        assert len(trials) == 1
        assert trials[0].state == optuna.trial.TrialState.WAITING


def test_template_trial_params_are_optuna_compatible() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/rustuna.journal"
        rustuna_storage = rustuna.storages.JournalFileStorage(file_path)
        rustuna_study = rustuna.create_study(
            storage=rustuna_storage, study_name="example"
        )
        rustuna_study.add_trial(
            rustuna.trial.create_trial(
                params={"x": 1.5, "n": 3},
                distributions={
                    "x": rustuna.distributions.FloatDistribution(0.0, 10.0),
                    "n": rustuna.distributions.IntDistribution(0, 10),
                },
                value=1.0,
            )
        )

        optuna_storage = JournalStorage(JournalFileBackend(file_path))
        study_id = optuna_storage.get_all_studies()[0]._study_id
        trials = optuna_storage.get_all_trials(study_id, deepcopy=False)
        assert len(trials) == 1
        assert trials[0].params == {"x": 1.5, "n": 3}
        assert isinstance(trials[0].params["n"], int)


def test_journal_file_storage_can_apply_discard() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/discarded.journal"
        storage = rustuna.storages.JournalFileStorage(file_path)
        study = rustuna.create_study(storage=storage, study_name="example")

        first = study.ask()
        second = study.ask()
        first_persisted = study.tell(first.number, 1.0)
        second_persisted = study.tell(second.number, 2.0)

        storage.discard_trials([first_persisted._trial_id])

        retained_trials = storage.get_trials(study._study_id)
        assert [trial._trial_id for trial in retained_trials] == [
            first_persisted._trial_id,
            second_persisted._trial_id,
        ]

        analysis_storage = rustuna.storages.JournalFileStorage(
            file_path,
            apply_discard=True,
        )
        omitted_trials = analysis_storage.get_trials(study._study_id)
        assert [trial._trial_id for trial in omitted_trials] == [
            second_persisted._trial_id,
        ]
        with pytest.raises(rustuna.exceptions.TrialDiscarded, match="Trial discarded"):
            analysis_storage.get_trial(first_persisted._trial_id)


def test_rustuna_journal_attrs_are_optuna_replayable() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = os.path.join(workdir, "attrs.journal")
        rustuna_storage = rustuna.storages.JournalFileStorage(file_path)
        rustuna_study = rustuna.create_study(
            storage=rustuna_storage, study_name="journal-attrs"
        )
        rustuna_study.set_user_attr("study_user", "study value")
        rustuna_storage.set_study_system_attrs(
            rustuna_study._study_id, {"study_system": "study value"}
        )

        trial = rustuna_study.ask()
        trial.set_user_attr("trial_user", "trial value")
        rustuna_storage.set_trial_system_attrs(
            trial._trial_id, {"trial_system": "trial value"}
        )
        rustuna_study.tell(trial.number, 1.0)

        with open(file_path) as f:
            logs = [json.loads(line) for line in f]

        study_user_log = next(log for log in logs if log["op_code"] == 2)
        assert study_user_log["user_attr"] == {"rustuna": None}
        assert study_user_log["user_attr_str"] == {"study_user": "study value"}

        study_system_log = next(log for log in logs if log["op_code"] == 3)
        assert study_system_log["system_attr"] == {"rustuna": None}
        assert study_system_log["system_attr_str"] == {"study_system": "study value"}

        trial_user_log = next(log for log in logs if log["op_code"] == 8)
        assert trial_user_log["user_attr"] == {"rustuna": None}
        assert trial_user_log["user_attr_str"] == {"trial_user": "trial value"}

        trial_system_log = next(log for log in logs if log["op_code"] == 9)
        assert trial_system_log["system_attr"] == {"rustuna": None}
        assert trial_system_log["system_attr_str"] == {"trial_system": "trial value"}

        optuna_storage = JournalStorage(JournalFileBackend(file_path))
        optuna_study = optuna.load_study(
            storage=optuna_storage, study_name="journal-attrs"
        )
        assert optuna_study.user_attrs == {"rustuna": None}
        assert optuna_storage.get_study_system_attrs(optuna_study._study_id) == {
            "rustuna": None
        }
        assert optuna_study.trials[0].user_attrs == {"rustuna": None}
        assert optuna_study.trials[0].system_attrs == {"rustuna": None}
