from __future__ import annotations

import tempfile
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import optuna
import pytest
from optuna.distributions import FloatDistribution
from optuna.exceptions import UpdateFinishedTrialError
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import StudyDirection
from optuna.testing.pytest_storages import StorageTestCase, _setup_studies
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState
from pytest import FixtureRequest

import rustuna
from rustuna.converter import ToOptunaStorage

if TYPE_CHECKING:
    from collections.abc import Generator

    from optuna.storages import BaseStorage


def _rustuna_storage(
    backend: str, workdir: str, attrs_format: Literal["json", "str"]
) -> rustuna.storages.StorageProtocol:
    if backend == "sqlite3":
        return rustuna.storages.SQLite3Storage(
            f"{workdir}/test.db",
            create_database=True,
            attrs_format=attrs_format,
        )
    return rustuna.storages.JournalFileStorage(
        f"{workdir}/test.journal", attrs_format=attrs_format
    )


def _optuna_storage(backend: str, workdir: str) -> BaseStorage:
    if backend == "sqlite3":
        return RDBStorage(f"sqlite:///{workdir}/test.db")
    return JournalStorage(JournalFileBackend(f"{workdir}/test.journal"))


@pytest.fixture(
    params=[
        ("sqlite3", "json"),
        ("journal-file", "json"),
        ("sqlite3", "str"),
        ("journal-file", "str"),
    ],
    ids=["sqlite3-json", "journal-file-json", "sqlite3-str", "journal-file-str"],
)
def storage(request: FixtureRequest) -> Generator[BaseStorage, None, None]:
    backend, attrs_format = request.param
    with tempfile.TemporaryDirectory() as workdir:
        yield ToOptunaStorage(_rustuna_storage(backend, workdir, attrs_format))


def _is_json_journal(storage: BaseStorage) -> bool:
    assert isinstance(storage, ToOptunaStorage)
    rustuna_storage = storage._storage
    return (
        isinstance(rustuna_storage, rustuna.storages.JournalFileStorage)
        and rustuna_storage.attrs_format == "json"
    )


class TestRustunaStorage(StorageTestCase):
    def test_get_all_studies(self, storage: BaseStorage) -> None:
        expected_frozen_studies, _ = _setup_studies(
            storage, n_study=10, n_trial=10, seed=46
        )
        frozen_studies = storage.get_all_studies()
        assert len(frozen_studies) == len(expected_frozen_studies)
        for _, expected_frozen_study in expected_frozen_studies.items():
            frozen_study = next(
                s
                for s in frozen_studies
                if s.study_name == expected_frozen_study.study_name
            )
            assert frozen_study.direction == expected_frozen_study.direction
            assert frozen_study.study_name == expected_frozen_study.study_name
            assert frozen_study.user_attrs == expected_frozen_study.user_attrs
            # Rustuna stores categorical choices as internal system attributes.
            system_attrs = {
                key: value
                for key, value in frozen_study.system_attrs.items()
                if not key.startswith("category_labels:")
            }
            assert system_attrs == expected_frozen_study.system_attrs

    def test_delete_study(self, storage: BaseStorage) -> None:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        storage.create_new_trial(study_id)
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 1

        # TODO(c-bata): Check study_id
        # with pytest.raises(KeyError):
        #     # Deletion of non-existent study.
        #     storage.delete_study(study_id + 1)

        storage.delete_study(study_id)
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 0

        # storage.delete_study(study_id)
        # with pytest.raises(KeyError):
        #     # Double free.
        #     storage.delete_study(study_id)

    def test_get_all_trials_uses_cache_diff(self, storage: BaseStorage) -> None:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id0 = storage.create_new_trial(study_id)
        trials1 = storage.get_all_trials(study_id)
        assert len(trials1) == 1
        assert {t._trial_id for t in trials1} == {trial_id0}

        trial_id1 = storage.create_new_trial(study_id)
        trials2 = storage.get_all_trials(study_id)
        assert len(trials2) == 2
        assert {t._trial_id for t in trials2} == {trial_id0, trial_id1}

    def test_set_and_get_study_user_attrs_for_floats(
        self, storage: BaseStorage
    ) -> None:
        if _is_json_journal(storage):
            pytest.skip(
                "Journal logs must be strict JSON, so NaN and Infinity are rejected."
            )
        super().test_set_and_get_study_user_attrs_for_floats(storage)

    def test_set_and_get_trial_user_attr_for_floats(self, storage: BaseStorage) -> None:
        if _is_json_journal(storage):
            pytest.skip(
                "Journal logs must be strict JSON, so NaN and Infinity are rejected."
            )
        super().test_set_and_get_trial_user_attr_for_floats(storage)

    def test_set_and_get_study_system_attrs_for_floats(
        self, storage: BaseStorage
    ) -> None:
        if _is_json_journal(storage):
            pytest.skip(
                "Journal logs must be strict JSON, so NaN and Infinity are rejected."
            )
        super().test_set_and_get_study_system_attrs_for_floats(storage)

    def test_set_and_get_trial_system_attr_for_floats(
        self, storage: BaseStorage
    ) -> None:
        if _is_json_journal(storage):
            pytest.skip(
                "Journal logs must be strict JSON, so NaN and Infinity are rejected."
            )
        super().test_set_and_get_trial_system_attr_for_floats(storage)

    @pytest.mark.skip("Rustuna cannot store objective values for failed state")
    def test_get_trial(self, storage: BaseStorage) -> None:
        super().test_get_trial(storage)

    @pytest.mark.skip("Rustuna cannot store objective values for failed state")
    def test_get_all_trials(self, storage: BaseStorage) -> None:
        super().test_get_all_trials(storage)

    @pytest.mark.skip("Rustuna's params cannot support the order of params")
    @pytest.mark.parametrize("param_names", [["a", "b"], ["b", "a"]])
    def test_get_all_trials_params_order(
        self, storage: BaseStorage, param_names: list[str]
    ) -> None: ...

    @pytest.mark.skip("Rustuna storages do not support pickle serialization")
    def test_pickle_storage(self, storage: BaseStorage) -> None: ...


@pytest.mark.parametrize("backend", ["sqlite3", "journal-file"])
def test_studies_written_through_rustuna_are_readable_by_optuna(backend: str) -> None:
    """With ``attrs_format="json"``, Optuna reads the same user attributes directly."""
    with tempfile.TemporaryDirectory() as workdir:
        rustuna_storage = _rustuna_storage(backend, workdir, "json")
        expected_studies, expected_trials = _setup_studies(
            ToOptunaStorage(rustuna_storage), n_study=3, n_trial=5, seed=1
        )
        del rustuna_storage

        optuna_storage = _optuna_storage(backend, workdir)
        studies = {s.study_name: s for s in optuna_storage.get_all_studies()}
        assert len(studies) == len(expected_studies)
        for study_id, expected_study in expected_studies.items():
            study = studies[expected_study.study_name]
            assert study.user_attrs == expected_study.user_attrs
            trials = optuna_storage.get_all_trials(study._study_id)
            expected = sorted(
                expected_trials[study_id].values(), key=lambda t: t.number
            )
            assert len(trials) == len(expected)
            for trial, expected_trial in zip(trials, expected):
                assert trial.number == expected_trial.number
                assert trial.state == expected_trial.state
                assert trial.params == expected_trial.params
                assert trial.user_attrs == expected_trial.user_attrs


@pytest.mark.parametrize("backend", ["sqlite3", "journal-file"])
def test_studies_written_by_optuna_are_readable_through_rustuna(backend: str) -> None:
    with tempfile.TemporaryDirectory() as workdir:
        expected_studies, expected_trials = _setup_studies(
            _optuna_storage(backend, workdir), n_study=3, n_trial=5, seed=2
        )

        storage = ToOptunaStorage(_rustuna_storage(backend, workdir, "json"))
        studies = {s.study_name: s for s in storage.get_all_studies()}
        assert len(studies) == len(expected_studies)
        for study_id, expected_study in expected_studies.items():
            study = studies[expected_study.study_name]
            assert study.user_attrs == expected_study.user_attrs
            trials = storage.get_all_trials(study._study_id)
            expected = sorted(
                expected_trials[study_id].values(), key=lambda t: t.number
            )
            assert [t.user_attrs for t in trials] == [t.user_attrs for t in expected]


def test_typed_user_attrs_are_shared_with_optuna_study() -> None:
    attrs = {"int": 1, "float": 0.5, "str": "123", "none": None, "list": [1, "a"]}
    with tempfile.TemporaryDirectory() as workdir:
        rustuna_study = rustuna.create_study(
            storage=_rustuna_storage("journal-file", workdir, "json"), study_name="s"
        )
        rustuna_study.set_user_attrs(attrs)
        trial = rustuna_study.ask()
        trial.set_user_attrs(attrs)
        rustuna_study.tell(trial.number, 1.0)

        optuna_study = optuna.load_study(
            study_name="s", storage=_optuna_storage("journal-file", workdir)
        )
        assert optuna_study.user_attrs == attrs
        assert optuna_study.trials[0].user_attrs == attrs
