#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AXIOMES BASE LABS
Module : Quantum State Fidelity
Version : 1.1.0

Fidélité entre états quantiques purs :
F(psi, phi) = |<psi|phi>|²

Infidélité :
I(psi, phi) = 1 - F(psi, phi)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_TOL = 1e-10

def _validate_state(
    state: ArrayLike,
    name: str,
    tol: float = DEFAULT_TOL,
) -> np.ndarray:
    """Valide un vecteur d'état quantique normalisé."""

    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol doit être fini et positif ou nul.")

    try:
        psi = np.asarray(state, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} contient des valeurs non numériques."
        ) from exc

    if psi.ndim != 1:
        raise ValueError(
            f"{name} doit être un vecteur unidimensionnel."
        )

    if psi.size == 0:
        raise ValueError(f"{name} ne peut pas être vide.")

    if not np.all(np.isfinite(psi)):
        raise ValueError(
            f"{name} contient NaN ou une valeur infinie."
        )

    # Norme calculée de façon stable numériquement.
    scale = np.max(np.abs(psi))

    if scale == 0:
        raise ValueError(
            f"{name} ne peut pas être le vecteur nul."
        )

    scaled_norm = np.linalg.norm(psi / scale)
    norm = scale * scaled_norm

    if not np.isfinite(norm) or norm == 0:
        raise ValueError(
            f"La norme de {name} est invalide."
        )

    if abs(norm - 1.0) > tol:
        raise ValueError(
            f"{name} doit être normalisé : "
            f"norme observée = {norm:.16g}."
        )

    return psi

def state_fidelity(
    psi_target: ArrayLike,
    psi_achieved: ArrayLike,
    tol: float = DEFAULT_TOL,
) -> float:
    """
    Calcule F = |<psi_target|psi_achieved>|².

    Les deux états doivent être normalisés et de même dimension.
    La phase globale ne modifie pas la fidélité.
    """

    target = _validate_state(
        psi_target, "psi_target", tol
    )
    achieved = _validate_state(
        psi_achieved, "psi_achieved", tol
    )

    if target.shape != achieved.shape:
        raise ValueError(
            "Les deux états doivent avoir la même dimension : "
            f"{target.size} != {achieved.size}."
        )

    overlap = np.vdot(target, achieved)
    fidelity = float(np.abs(overlap) ** 2)

    # Corrige uniquement les écarts d'arrondi près de 1.
    if 1.0 < fidelity <= 1.0 + 100 * np.finfo(float).eps:
        fidelity = 1.0

    if not 0.0 <= fidelity <= 1.0:
        raise ArithmeticError(
            f"Fidélité hors limites physiques : {fidelity}."
        )

    return fidelity

def infidelity(
    psi_target: ArrayLike,
    psi_achieved: ArrayLike,
    tol: float = DEFAULT_TOL,
) -> float:
    """
    Calcule I = 1 - F.

    I = 0 : états identiques à une phase globale près.
    I = 1 : états orthogonaux.
    """

    return 1.0 - state_fidelity(
        psi_target,
        psi_achieved,
        tol=tol,
    )

def run_validation_tests() -> None:
    """Exécute des tests numériques de base."""

    psi = np.array(
        [1 / np.sqrt(2), 1j / np.sqrt(2)],
        dtype=np.complex128,
    )

    # Même état : fidélité 1.
    assert np.isclose(state_fidelity(psi, psi), 1.0)

    # Phase globale : fidélité 1.
    assert np.isclose(
        state_fidelity(psi, np.exp(1j * 0.7) * psi),
        1.0,
    )

    # États orthogonaux : fidélité 0.
    orthogonal = np.array(
        [1 / np.sqrt(2), -1j / np.sqrt(2)]
    )
    assert np.isclose(
        state_fidelity(psi, orthogonal), 0.0
    )

    # Infidélité d'un état avec lui-même : 0.
    assert np.isclose(infidelity(psi, psi), 0.0)

    # Les entrées invalides doivent être rejetées.
    invalid_cases = [
        (np.array([1.0, 1.0]), "non normalisé"),
        (np.array([1.0, np.nan]), "NaN"),
        (np.array([0.0, 0.0]), "vecteur nul"),
        (np.array([[1.0, 0.0]]), "dimension"),
    ]

    for invalid_state, label in invalid_cases:
        try:
            state_fidelity(invalid_state, psi)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Le test aurait dû échouer : {label}"
            )

    print("VALIDATION : PASS")
    print(f"Fidélité identique : {state_fidelity(psi, psi):.12f}")
    print(f"Infidélité identique : {infidelity(psi, psi):.12e}")
    print(
        "Fidélité états orthogonaux : "
        f"{state_fidelity(psi, orthogonal):.12f}"
    )

if __name__ == "__main__":
    run_validation_tests()
