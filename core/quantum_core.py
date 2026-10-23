#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AXIOMES BASE LABS — QuantumLab
Module : quantum_core.py
Version : 0.1.0

Noyau dynamique quantique.

Fonctions :
    - Validation des états quantiques
    - Validation des Hamiltoniens hermitiens
    - Évolution temporelle par solve_ivp
    - Contrôle de conservation de la norme
    - Hamiltonien paramétrique à deux niveaux
    - Tests analytiques et numériques

Convention :
    i*hbar*d|psi>/dt = H(t)|psi>

Dépendances :
    NumPy
    SciPy

Cette version traite les états purs et les systèmes fermés.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp


ComplexArray = NDArray[np.complex128]
Hamiltonian = Callable[[float], ArrayLike]


@dataclass(frozen=True)
class EvolutionResult:
    """Résultats d'une évolution quantique."""

    times: NDArray[np.float64]
    states: ComplexArray
    norm_errors: NDArray[np.float64]
    solver_message: str
    function_evaluations: int

    @property
    def final_state(self) -> ComplexArray:
        """État final calculé."""
        return self.states[-1].copy()

    @property
    def max_norm_error(self) -> float:
        """Écart maximal absolu à la norme unité."""
        return float(np.max(self.norm_errors))


def validate_state(
    psi: ArrayLike,
    tolerance: float = 1e-10,
) -> ComplexArray:
    """Valide un état pur normalisé."""

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance doit être finie et positive ou nulle.")

    try:
        state = np.asarray(psi, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("État non numérique.") from exc

    if state.ndim != 1 or state.size == 0:
        raise ValueError("L'état doit être un vecteur non vide.")

    if not np.all(np.isfinite(state)):
        raise ValueError("L'état contient des valeurs non finies.")

    norm = float(np.linalg.norm(state))

    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Norme de l'état invalide.")

    if abs(norm - 1.0) > tolerance:
        raise ValueError(
            f"État non normalisé : norme={norm:.16g}."
        )

    return state.copy()


def validate_hamiltonian(
    H: ArrayLike,
    dimension: int | None = None,
    tolerance: float = 1e-10,
) -> ComplexArray:
    """Valide la dimension, la finitude et l'hermiticité de H."""

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolérance d'hermiticité invalide.")

    try:
        matrix = np.asarray(H, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Hamiltonien non numérique.") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Le Hamiltonien doit être une matrice carrée.")

    if matrix.shape[0] == 0:
        raise ValueError("Le Hamiltonien ne peut pas être vide.")

    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ValueError(
            f"Dimension incompatible : {matrix.shape}, "
            f"attendu {(dimension, dimension)}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Le Hamiltonien contient NaN ou Inf.")

    if not np.allclose(
        matrix,
        matrix.conj().T,
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("Le Hamiltonien n'est pas hermitien.")

    return matrix


def qubit_hamiltonian(
    delta: float,
    omega_x: float,
    omega_y: float = 0.0,
    hbar: float = 1.0,
) -> ComplexArray:
    """
    Hamiltonien à deux niveaux.

    H = hbar/2 * [[ delta, omega_x - i*omega_y ],
                  [ omega_x + i*omega_y, -delta ]]

    delta, omega_x et omega_y sont des fréquences angulaires.
    hbar est exprimée dans les unités cohérentes avec H.
    """

    values = np.asarray(
        [delta, omega_x, omega_y, hbar],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("Paramètres du qubit non finis.")

    if hbar <= 0.0:
        raise ValueError("hbar doit être strictement positif.")

    matrix = 0.5 * hbar * np.array(
        [
            [delta, omega_x - 1j * omega_y],
            [omega_x + 1j * omega_y, -delta],
        ],
        dtype=np.complex128,
    )

    return validate_hamiltonian(matrix)


def evolve(
    psi0: ArrayLike,
    hamiltonian: Hamiltonian,
    t_span: tuple[float, float],
    t_eval: ArrayLike,
    *,
    hbar: float = 1.0,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    hermitian_tolerance: float = 1e-10,
    norm_tolerance: float = 1e-7,
) -> EvolutionResult:
    """
    Calcule l'évolution d'un état pur dans un système fermé.

    Le Hamiltonien est une fonction H(t), exprimée en unités
    d'énergie cohérentes avec hbar et l'unité de temps.

    norm_tolerance contrôle l'écart maximal admis à la norme
    unité sur les points retournés. Un dépassement déclenche
    une RuntimeError plutôt qu'une correction silencieuse.
    """

    psi0 = validate_state(psi0)

    if not callable(hamiltonian):
        raise TypeError("hamiltonian doit être appelable.")

    hbar = float(hbar)
    rtol = float(rtol)
    atol = float(atol)
    hermitian_tolerance = float(hermitian_tolerance)
    norm_tolerance = float(norm_tolerance)

    for name, value in (
        ("hbar", hbar),
        ("rtol", rtol),
        ("atol", atol),
        ("hermitian_tolerance", hermitian_tolerance),
        ("norm_tolerance", norm_tolerance),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} doit être fini.")

    if hbar <= 0 or rtol <= 0 or atol <= 0:
        raise ValueError(
            "hbar, rtol et atol doivent être strictement positifs."
        )

    if hermitian_tolerance < 0 or norm_tolerance < 0:
        raise ValueError("Les tolérances doivent être positives ou nulles.")

    try:
        t_start, t_end = map(float, t_span)
    except (TypeError, ValueError) as exc:
        raise ValueError("t_span doit contenir deux temps.") from exc

    if not np.isfinite(t_start) or not np.isfinite(t_end):
        raise ValueError("Les bornes temporelles doivent être finies.")

    if t_end <= t_start:
        raise ValueError("Il faut t_end > t_start.")

    times = np.asarray(t_eval, dtype=float)

    if times.ndim != 1 or times.size == 0:
        raise ValueError("t_eval doit être un vecteur non vide.")

    if not np.all(np.isfinite(times)):
        raise ValueError("t_eval contient des valeurs non finies.")

    if np.any(np.diff(times) <= 0):
        raise ValueError("t_eval doit être strictement croissant.")

    if times[0] < t_start or times[-1] > t_end:
        raise ValueError("t_eval doit rester dans t_span.")

    dimension = psi0.size

    def rhs(t: float, psi: ComplexArray) -> ComplexArray:
        H = validate_hamiltonian(
            hamiltonian(t),
            dimension=dimension,
            tolerance=hermitian_tolerance,
        )

        return -1j * (H @ psi) / hbar

    solution = solve_ivp(
        rhs,
        (t_start, t_end),
        psi0,
        t_eval=times,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(
            f"Échec de l'intégration : {solution.message}"
        )

    states = np.asarray(solution.y.T, dtype=np.complex128)

    if not np.all(np.isfinite(states)):
        raise FloatingPointError(
            "L'intégration a produit des valeurs non finies."
        )

    norms = np.linalg.norm(states, axis=1)
    norm_errors = np.abs(norms - 1.0)
    max_error = float(np.max(norm_errors))

    if max_error > norm_tolerance:
        raise RuntimeError(
            "Conservation de la norme hors tolérance : "
            f"erreur maximale={max_error:.3e}, "
            f"limite={norm_tolerance:.3e}."
        )

    return EvolutionResult(
        times=times.copy(),
        states=states,
        norm_errors=norm_errors,
        solver_message=solution.message,
        function_evaluations=solution.nfev,
    )


def run_validation_tests() -> None:
    """Tests de référence du noyau dynamique."""

    print("AXIOMES BASE LABS — QuantumLab")
    print("Tests du quantum_core.py")
    print("-" * 48)

    omega = 2.0
    times = np.linspace(0.0, 2.0, 1001)
    psi0 = np.array([1.0, 0.0], dtype=complex)

    # H constant : rotation cohérente autour de l'axe X.
    def H_of_t(t: float) -> ComplexArray:
        return qubit_hamiltonian(
            delta=0.0,
            omega_x=omega,
        )

    result = evolve(
        psi0,
        H_of_t,
        (times[0], times[-1]),
        times,
    )

    # Solution analytique pour H = hbar*omega*sigma_x/2.
    exact = np.column_stack(
        [
            np.cos(omega * times / 2.0),
            -1j * np.sin(omega * times / 2.0),
        ]
    )

    state_error = np.max(
        np.linalg.norm(result.states - exact, axis=1)
    )

    assert result.max_norm_error < 1e-7
    assert state_error < 1e-7

    # Le Hamiltonien doit être hermitien.
    try:
        validate_hamiltonian(
            np.array([[0.0, 1.0], [0.0, 0.0]])
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Hamiltonien non hermitien accepté.")

    # L'état initial doit être normalisé.
    try:
        validate_state([1.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("État non normalisé accepté.")

    print("TEST 1 — Évolution analytique : PASS")
    print(f"Erreur maximale d'état : {state_error:.3e}")
    print("TEST 2 — Conservation de la norme : PASS")
    print(f"Erreur maximale de norme : {result.max_norm_error:.3e}")
    print("TEST 3 — Rejet Hamiltonien non hermitien : PASS")
    print("TEST 4 — Rejet état non normalisé : PASS")
    print(f"Évaluations du Hamiltonien : {result.function_evaluations}")
    print("RÉSULTAT : VALIDATION PASSÉE")


if __name__ == "__main__":
    run_validation_tests()
