#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXIOMES BASE LABS — Quantum Core
Module : validation.py
Version : 0.1.0

Validation indépendante du moteur quantique :
1. Solution analytique d'un qubit
2. Convergence sous resserrement des tolérances
3. Invariance de phase globale
4. Conservation de la norme
5. Rejet des états et Hamiltoniens invalides
6. Contrat explicite des unités

Dépendances :
    numpy
    scipy
    quantum_core.py

Exécution :
    python validation.py
"""

from __future__ import annotations

import sys
import numpy as np

from quantum_core import (
    evolve,
    qubit_hamiltonian,
    validate_state,
    validate_hamiltonian,
)


# ============================================================
# Configuration
# ============================================================

EPS = np.finfo(float).eps

BASE_RTOL = 1e-8
BASE_ATOL = 1e-10

STRICT_RTOL = 1e-12
STRICT_ATOL = 1e-14

ANALYTIC_ERROR_LIMIT = 1e-8
CONVERGENCE_ERROR_LIMIT = 1e-8
PHASE_ERROR_LIMIT = 1e-9
NORM_ERROR_LIMIT = 1e-7


# ============================================================
# Infrastructure de test
# ============================================================

class ValidationReport:
    """Collecte les résultats sans interrompre les tests."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.results: list[tuple[str, bool, str]] = []

    def check(
        self,
        name: str,
        condition: bool,
        details: str = "",
    ) -> None:
        success = bool(condition)

        if success:
            self.passed += 1
            status = "PASS"
        else:
            self.failed += 1
            status = "FAIL"

        self.results.append((name, success, details))

        suffix = f" — {details}" if details else ""
        print(f"[{status}] {name}{suffix}")

    def summary(self) -> int:
        total = self.passed + self.failed

        print("\n" + "=" * 60)
        print("RAPPORT DE VALIDATION — QUANTUM CORE")
        print("=" * 60)
        print(f"Tests réussis : {self.passed}/{total}")
        print(f"Tests échoués : {self.failed}/{total}")

        if self.failed:
            print("STATUT GLOBAL : ÉCHEC")
            return 1

        print("STATUT GLOBAL : PASS")
        return 0


def max_state_error(
    states_a: np.ndarray,
    states_b: np.ndarray,
) -> float:
    """Erreur vectorielle maximale sur l'ensemble des temps."""
    return float(np.max(np.linalg.norm(states_a - states_b, axis=1)))


def pure_state_fidelity(
    psi_a: np.ndarray,
    psi_b: np.ndarray,
) -> float:
    """
    Fidélité entre deux états purs normalisés :
        F = |<psi_a | psi_b>|²
    """
    overlap = np.vdot(psi_a, psi_b)
    return float(np.abs(overlap) ** 2)


# ============================================================
# 1. Solution analytique
# ============================================================

def test_analytic_solution(report: ValidationReport) -> None:
    """
    Hamiltonien :
        H = (hbar * omega / 2) * sigma_x

    État initial :
        |psi(0)> = |0> = [1, 0]

    Solution exacte :
        |psi(t)> = [
            cos(omega*t/2),
            -i*sin(omega*t/2)
        ]
    """

    hbar = 1.0
    omega = 2.0

    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    t_start = 0.0
    t_end = 2.0
    times = np.linspace(t_start, t_end, 401)

    H = qubit_hamiltonian(
        delta=0.0,
        omega_x=omega,
        omega_y=0.0,
        hbar=hbar,
    )

    result = evolve(
        psi0=psi0,
        hamiltonian=lambda t: H,
        t_span=(t_start, t_end),
        t_eval=times,
        hbar=hbar,
        rtol=BASE_RTOL,
        atol=BASE_ATOL,
    )

    exact = np.column_stack((
        np.cos(omega * times / 2.0),
        -1j * np.sin(omega * times / 2.0),
    ))

    error = max_state_error(result.states, exact)

    report.check(
        "Solution analytique du qubit",
        error < ANALYTIC_ERROR_LIMIT,
        f"erreur maximale={error:.3e}",
    )

    report.check(
        "Conservation de la norme",
        result.max_norm_error < NORM_ERROR_LIMIT,
        f"erreur maximale de norme={result.max_norm_error:.3e}",
    )


# ============================================================
# 2. Convergence numérique
# ============================================================

def test_convergence(report: ValidationReport) -> None:
    """
    Compare deux intégrations :
      A. tolérances standard
      B. tolérances resserrées

    Les deux trajectoires sont également comparées à la solution
    analytique afin d'éviter de confondre accord numérique et exactitude.
    """

    hbar = 1.0
    omega = 2.0

    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    t_start = 0.0
    t_end = 2.0
    times = np.linspace(t_start, t_end, 401)

    H = qubit_hamiltonian(
        delta=0.0,
        omega_x=omega,
        omega_y=0.0,
        hbar=hbar,
    )

    standard = evolve(
        psi0=psi0,
        hamiltonian=lambda t: H,
        t_span=(t_start, t_end),
        t_eval=times,
        hbar=hbar,
        rtol=BASE_RTOL,
        atol=BASE_ATOL,
    )

    strict = evolve(
        psi0=psi0,
        hamiltonian=lambda t: H,
        t_span=(t_start, t_end),
        t_eval=times,
        hbar=hbar,
        rtol=STRICT_RTOL,
        atol=STRICT_ATOL,
    )

    exact = np.column_stack((
        np.cos(omega * times / 2.0),
        -1j * np.sin(omega * times / 2.0),
    ))

    error_standard = max_state_error(standard.states, exact)
    error_strict = max_state_error(strict.states, exact)
    difference = max_state_error(standard.states, strict.states)

    report.check(
        "Précision avec tolérances standard",
        error_standard < CONVERGENCE_ERROR_LIMIT,
        f"erreur analytique={error_standard:.3e}",
    )

    report.check(
        "Précision avec tolérances strictes",
        error_strict < CONVERGENCE_ERROR_LIMIT,
        f"erreur analytique={error_strict:.3e}",
    )

    report.check(
        "Accord entre les deux intégrations",
        difference < CONVERGENCE_ERROR_LIMIT,
        f"écart maximal={difference:.3e}",
    )

    report.check(
        "Amélioration ou stabilité de la précision",
        error_strict <= error_standard + 100.0 * EPS,
        (
            f"standard={error_standard:.3e}, "
            f"strict={error_strict:.3e}"
        ),
    )


# ============================================================
# 3. Invariance de phase globale
# ============================================================

def test_global_phase_invariance(
    report: ValidationReport,
) -> None:
    """
    Compare psi et exp(i*theta)*psi.

    La phase globale ne doit modifier ni les probabilités
    de mesure ni la fidélité physique entre les états.
    """

    hbar = 1.0
    omega = 1.3
    theta = 0.731

    psi0 = np.array(
        [1.0 / np.sqrt(2.0), 1j / np.sqrt(2.0)],
        dtype=complex,
    )

    psi0_phase = np.exp(1j * theta) * psi0

    times = np.linspace(0.0, 3.0, 201)

    H = qubit_hamiltonian(
        delta=0.4,
        omega_x=omega,
        omega_y=0.2,
        hbar=hbar,
    )

    result_a = evolve(
        psi0=psi0,
        hamiltonian=lambda t: H,
        t_span=(0.0, 3.0),
        t_eval=times,
        hbar=hbar,
    )

    result_b = evolve(
        psi0=psi0_phase,
        hamiltonian=lambda t: H,
        t_span=(0.0, 3.0),
        t_eval=times,
        hbar=hbar,
    )

    expected_b = np.exp(1j * theta) * result_a.states

    trajectory_error = max_state_error(
        result_b.states,
        expected_b,
    )

    report.check(
        "Covariance de la trajectoire sous phase globale",
        trajectory_error < PHASE_ERROR_LIMIT,
        f"erreur maximale={trajectory_error:.3e}",
    )

    fidelity = pure_state_fidelity(
        result_a.final_state,
        result_b.final_state,
    )

    report.check(
        "Invariance de la fidélité",
        abs(fidelity - 1.0) < PHASE_ERROR_LIMIT,
        f"fidélité={fidelity:.15f}",
    )

    probabilities_a = np.abs(result_a.states) ** 2
    probabilities_b = np.abs(result_b.states) ** 2

    probability_error = float(
        np.max(np.abs(probabilities_a - probabilities_b))
    )

    report.check(
        "Invariance des probabilités de mesure",
        probability_error < PHASE_ERROR_LIMIT,
        f"écart maximal={probability_error:.3e}",
    )


# ============================================================
# 4. Validation des entrées
# ============================================================

def test_invalid_inputs(report: ValidationReport) -> None:
    """Vérifie le rejet des entrées physiquement ou numériquement invalides."""

    try:
        validate_state(np.array([2.0, 0.0], dtype=complex))
        rejected = False
    except (ValueError, TypeError):
        rejected = True

    report.check(
        "Rejet d'un état non normalisé",
        rejected,
    )

    try:
        validate_state(np.array([1.0, np.nan], dtype=complex))
        rejected = False
    except (ValueError, TypeError):
        rejected = True

    report.check(
        "Rejet d'un état non fini",
        rejected,
    )

    non_hermitian = np.array(
        [[0.0, 1.0],
         [0.0, 0.0]],
        dtype=complex,
    )

    try:
        validate_hamiltonian(non_hermitian)
        rejected = False
    except (ValueError, TypeError):
        rejected = True

    report.check(
        "Rejet d'un Hamiltonien non hermitien",
        rejected,
    )

    valid_hamiltonian = np.eye(2, dtype=complex)

    try:
        validate_hamiltonian(valid_hamiltonian, dimension=3)
        rejected = False
    except (ValueError, TypeError):
        rejected = True

    report.check(
        "Rejet d'une dimension incompatible",
        rejected,
    )


# ============================================================
# 5. Contrat des unités
# ============================================================

def test_unit_contract(report: ValidationReport) -> None:
    """
    Un contrôle numérique ne peut pas déduire les unités physiques
    à partir de simples tableaux de nombres.

    Ce test documente la convention du moteur :
        i*hbar*d|psi>/dt = H|psi>

    Si H est exprimé en joules :
        hbar doit être en joule-seconde
        t doit être en secondes.

    Si hbar = 1, le modèle utilise des unités naturelles cohérentes.
    """

    hbar = 1.0
    omega = 2.0
    times = np.linspace(0.0, 1.0, 11)

    H = qubit_hamiltonian(
        delta=0.0,
        omega_x=omega,
        hbar=hbar,
    )

    # Contrôle dimensionnel algébrique :
    # H/hbar possède l'échelle numérique d'une fréquence angulaire.
    angular_frequency_scale = np.linalg.norm(H, ord=2) / hbar

    report.check(
        "Hamiltonien fini et hermitien",
        np.all(np.isfinite(H))
        and np.allclose(H, H.conj().T),
    )

    report.check(
        "Échelle H/hbar finie et positive",
        np.isfinite(angular_frequency_scale)
        and angular_frequency_scale > 0.0,
        f"échelle numérique={angular_frequency_scale:.6g}",
    )

    report.check(
        "Échelle temporelle finie et croissante",
        np.all(np.isfinite(times))
        and np.all(np.diff(times) > 0.0),
    )

    print(
        "[INFO] Cohérence physique des unités : convention à documenter. "
        "Elle ne peut pas être prouvée à partir de nombres sans unités."
    )


# ============================================================
# Point d'entrée
# ============================================================

def main() -> int:
    report = ValidationReport()

    print("=" * 60)
    print("AXIOMES BASE LABS — VALIDATION DU QUANTUM CORE")
    print("=" * 60)

    test_analytic_solution(report)
    test_convergence(report)
    test_global_phase_invariance(report)
    test_invalid_inputs(report)
    test_unit_contract(report)

    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
