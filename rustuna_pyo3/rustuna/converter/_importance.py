from __future__ import annotations

from collections.abc import Callable

import optuna
from optuna.importance import BaseImportanceEvaluator
from optuna.trial import FrozenTrial

import rustuna
from rustuna.converter._storage import ToRustunaStorage
from rustuna.converter._trial import to_frozen_trial


class ToOptunaImportanceEvaluator(BaseImportanceEvaluator):
    def __init__(
        self,
        evaluator: rustuna.importance.PedAnovaImportanceEvaluator,
    ) -> None:
        self._evaluator = evaluator

    def evaluate(
        self,
        study: optuna.Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        rustuna_study = rustuna.load_study(
            study_name=study.study_name, storage=ToRustunaStorage(study._storage)
        )
        rustuna_target = None if target is None else _to_rustuna_target(target)
        return self._evaluator.evaluate(
            rustuna_study, params=params, target=rustuna_target
        )


def _to_rustuna_target(
    target: Callable[[FrozenTrial], float],
) -> Callable[[rustuna.trial.PersistedTrial], float]:
    return lambda trial: target(to_frozen_trial(trial))
