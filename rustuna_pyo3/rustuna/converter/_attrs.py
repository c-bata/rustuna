from __future__ import annotations

import json
from typing import Any

# Optuna exposes attrs as `dict[str, Any]`, while Rustuna stores them as
# `dict[str, str]`. String values are stored under their original keys so that
# Rustuna-native and Optuna-originated string attrs have the same representation.
# Non-string Optuna values are JSON-encoded under their original keys and are
# accompanied by a marker key. The marker lets `to_optuna_attrs` restore the
# original type without attempting to JSON-decode ordinary Rustuna strings.
# Constraint attrs are the exception: Rustuna stores them as strings, while
# Optuna exposes constraint values as floats.
#
# The marker key is an implementation detail and is omitted when converting
# attrs back to Optuna. Rustuna may also store internal system attrs, such as
# categorical labels; callers exposing attrs through Optuna are responsible for
# filtering those Rustuna-specific attributes when necessary.
_PREFIX_REQUIRE_JSON_ENCODE = "optuna_json_encoded:"
_CONSTRAINTS_KEY = "constraints"
_CONSTRAINTS_PREFIX = f"{_CONSTRAINTS_KEY}:"


def to_optuna_attrs(attrs: dict[str, str]) -> dict[str, Any]:
    converted = {}
    for key, value in attrs.items():
        if key.startswith(_PREFIX_REQUIRE_JSON_ENCODE):
            continue
        if _PREFIX_REQUIRE_JSON_ENCODE + key in attrs:
            converted[key] = json.loads(value)
        elif key.startswith(_CONSTRAINTS_PREFIX):
            # Convert Rustuna constraints to Optuna constraints
            converted[key] = float(value)
        else:
            converted[key] = value
    return converted


def to_rustuna_json_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Convert Optuna's attrs for Rustuna storages with ``attrs_format="json"``.

    Such storages accept any JSON-serializable value, so the values are passed as they are.
    Only Optuna's legacy ``constraints`` list is split into Rustuna's constraint keys.
    """
    converted = dict(attrs)
    if _CONSTRAINTS_KEY in converted:
        constraints = converted.pop(_CONSTRAINTS_KEY)
        for i, c in enumerate(constraints):
            converted[_CONSTRAINTS_PREFIX + str(i)] = c
    return converted


def to_rustuna_attrs(attrs: dict[str, Any]) -> dict[str, str]:
    converted = {}
    for key, value in attrs.items():
        if isinstance(value, str):
            converted[key] = value
            continue
        converted[key] = json.dumps(value)
        converted[_PREFIX_REQUIRE_JSON_ENCODE + key] = "true"

    # Convert Optuna's legacy constraint to Rustuna constraints
    if _CONSTRAINTS_KEY in converted:
        constraints = json.loads(converted.pop(_CONSTRAINTS_KEY))
        for i, c in enumerate(constraints):
            converted[_CONSTRAINTS_PREFIX + str(i)] = str(c)
    return converted
