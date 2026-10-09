from __future__ import annotations

from typing import Literal

from optuna.study._frozen import FrozenStudy

import rustuna

from ._attrs import to_optuna_attrs
from ._direction import to_optuna_directions, to_rustuna_directions
from ._trial import convert_attrs_to_rustuna


def to_frozen_study(study: rustuna.study.PersistedStudy) -> FrozenStudy:
    """Convert a Rustuna persisted study to an Optuna frozen study."""
    return FrozenStudy(
        study_name=study.name,
        study_id=study.id,
        direction=None,
        directions=to_optuna_directions(study.directions),
        user_attrs=to_optuna_attrs(study.user_attrs),
        system_attrs=to_optuna_attrs(study.system_attrs),
    )


def to_persisted_study(
    study: FrozenStudy,
    *,
    attrs_format: Literal["marker", "json"] = "marker",
) -> rustuna.study.PersistedStudy:
    """Convert an Optuna frozen study to a Rustuna persisted study.

    See :func:`rustuna.converter.to_persisted_trial` for ``attrs_format``.
    """
    user_attrs, system_attrs = convert_attrs_to_rustuna(
        study.user_attrs, study.system_attrs, attrs_format
    )
    return rustuna.study.PersistedStudy(
        id=study._study_id,
        name=study.study_name,
        directions=to_rustuna_directions(study.directions),
        user_attrs=user_attrs,
        system_attrs=system_attrs,
        attrs_format="json" if attrs_format == "json" else "str",
    )
