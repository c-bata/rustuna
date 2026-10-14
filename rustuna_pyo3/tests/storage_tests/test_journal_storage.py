from __future__ import annotations

import json
import os
import tempfile
from typing import Literal

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


@pytest.mark.parametrize("attrs_format", ["json", "str"])
def test_rustuna_journal_attrs_are_optuna_replayable(
    attrs_format: Literal["json", "str"],
) -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = os.path.join(workdir, "attrs.journal")
        rustuna_storage = rustuna.storages.JournalFileStorage(
            file_path, attrs_format=attrs_format
        )
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
        trial_user_log = next(log for log in logs if log["op_code"] == 8)
        if attrs_format == "json":
            # Optuna's schema.
            assert study_user_log["user_attr"] == {"study_user": "study value"}
            assert "user_attr_str" not in study_user_log
            assert trial_user_log["user_attr"] == {"trial_user": "trial value"}
            assert "user_attr_str" not in trial_user_log
        else:
            assert study_user_log["user_attr"] == {"rustuna": None}
            assert study_user_log["user_attr_str"] == {"study_user": "study value"}
            assert trial_user_log["user_attr"] == {"rustuna": None}
            assert trial_user_log["user_attr_str"] == {"trial_user": "trial value"}

        study_system_log = next(log for log in logs if log["op_code"] == 3)
        trial_system_log = next(log for log in logs if log["op_code"] == 9)
        if attrs_format == "json":
            assert study_system_log["system_attr"] == {"study_system": "study value"}
            assert "system_attr_str" not in study_system_log
            assert trial_system_log["system_attr"] == {"trial_system": "trial value"}
            assert "system_attr_str" not in trial_system_log
        else:
            assert study_system_log["system_attr"] == {"rustuna": None}
            assert study_system_log["system_attr_str"] == {
                "study_system": "study value"
            }
            assert trial_system_log["system_attr"] == {"rustuna": None}
            assert trial_system_log["system_attr_str"] == {
                "trial_system": "trial value"
            }

        optuna_storage = JournalStorage(JournalFileBackend(file_path))
        optuna_study = optuna.load_study(
            storage=optuna_storage, study_name="journal-attrs"
        )
        expected_study_user_attrs = (
            {"study_user": "study value"}
            if attrs_format == "json"
            else {"rustuna": None}
        )
        expected_trial_user_attrs = (
            {"trial_user": "trial value"}
            if attrs_format == "json"
            else {"rustuna": None}
        )
        expected_study_system_attrs = (
            {"study_system": "study value"}
            if attrs_format == "json"
            else {"rustuna": None}
        )
        expected_trial_system_attrs = (
            {"trial_system": "trial value"}
            if attrs_format == "json"
            else {"rustuna": None}
        )
        assert optuna_study.user_attrs == expected_study_user_attrs
        assert (
            optuna_storage.get_study_system_attrs(optuna_study._study_id)
            == expected_study_system_attrs
        )
        assert optuna_study.trials[0].user_attrs == expected_trial_user_attrs
        assert optuna_study.trials[0].system_attrs == expected_trial_system_attrs


@pytest.mark.parametrize("attrs_format", ["json", "str"])
def test_optuna_journal_user_attrs_are_readable(
    attrs_format: Literal["json", "str"],
) -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = os.path.join(workdir, "attrs.journal")
        optuna_study = optuna.create_study(
            storage=JournalStorage(JournalFileBackend(file_path)),
            study_name="optuna-attrs",
        )
        optuna_study.set_user_attr("elements", ["O", "Ti"])
        optuna_study.set_user_attr("name", "TiO2")
        optuna_study.add_trial(
            optuna.trial.create_trial(
                value=1.0, user_attrs={"generation": 3, "formula": "TiO2"}
            )
        )
        optuna_trial = optuna_study.ask()
        optuna_trial.set_user_attr("opt_stats", {"n_force_calls": 5})
        optuna_study.tell(optuna_trial, 2.0)

        rustuna_study = rustuna.load_study(
            storage=rustuna.storages.JournalFileStorage(
                file_path, attrs_format=attrs_format
            ),
            study_name="optuna-attrs",
        )
        trials = rustuna_study.get_trials()
        if attrs_format == "json":
            assert rustuna_study.user_attrs == {"elements": ["O", "Ti"], "name": "TiO2"}
            assert trials[0].user_attrs == {"generation": 3, "formula": "TiO2"}
            assert trials[1].user_attrs == {"opt_stats": {"n_force_calls": 5}}
        else:
            # Strings are unquoted and other values are exposed as JSON texts.
            # Optuna's journal writes compact JSON.
            assert rustuna_study.user_attrs == {
                "elements": '["O","Ti"]',
                "name": "TiO2",
            }
            assert trials[0].user_attrs == {"generation": "3", "formula": "TiO2"}
            assert trials[1].user_attrs == {"opt_stats": '{"n_force_calls":5}'}
