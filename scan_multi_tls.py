#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXIOMES BASE LABS
Time-Reversal Noise Witness — Qubit + TLS
Version: 2.0.0

Modèle :
    - Deux qubits système + un TLS environnemental (dimension 8).
    - Matrices de densité et équation maîtresse de Lindblad.
    - Bruits classiques 1/f et radiofréquence reproductibles.
    - Comparaison appariée de profils temporels.
    - Contrôle de réversibilité unitaire sans dissipation.
    - Diagnostics : trace, hermiticité, positivité, hermiticité de H.
    - Export JSON et figure.

Conventions :
    hbar = 1
    Le taux gamma_TLS est le taux de relaxation de population.
    L_TLS = sqrt(gamma_TLS) |g><e|.
    Le TLS initial est dans son état fondamental |g> = |0>.

Dépendances :
    numpy, scipy, matplotlib

Exécution :
    python time_reversal_noise_witness.py

Attention :
    Une comparaison de fidélités ne constitue pas, à elle seule,
    une preuve de non-markovianité ou d'irréversibilité fondamentale.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp


# ============================================================
# JOURNALISATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger("AXIOMES.TimeReversal")


ComplexMatrix = np.ndarray
Hamiltonian = Callable[[float], ComplexMatrix]


# ============================================================
# OPÉRATEURS DE BASE
# ============================================================

SX = np.array(
    [[0, 1], [1, 0]],
    dtype=np.complex128,
)

SY = np.array(
    [[0, -1j], [1j, 0]],
    dtype=np.complex128,
)

SZ = np.array(
    [[1, 0], [0, -1]],
    dtype=np.complex128,
)

I2 = np.eye(2, dtype=np.complex128)
I4 = np.eye(4, dtype=np.complex128)
I8 = np.eye(8, dtype=np.complex128)

# Sous-espace à une excitation :
# |00>, |01>, |10>, |11>
#
# SX_SUB échange |01> et |10>.
# SY_SUB est la composante de Pauli Y correspondante.
# SZ_SUB attribue +1 à |01> et -1 à |10>.

SX_SUB = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ],
    dtype=np.complex128,
)

SY_SUB = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, -1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 0],
    ],
    dtype=np.complex128,
)

SZ_SUB = np.diag(
    [0, 1, -1, 0]
).astype(np.complex128)

# Ordre tensoriel :
# qubit 1 ⊗ qubit 2 ⊗ TLS

TX = np.kron(I4, SX)
TY = np.kron(I4, SY)
TZ = np.kron(I4, SZ)

# États de base du système à deux qubits.
KET00 = np.array([1, 0, 0, 0], dtype=np.complex128)
KET01 = np.array([0, 1, 0, 0], dtype=np.complex128)
KET10 = np.array([0, 0, 1, 0], dtype=np.complex128)
KET11 = np.array([0, 0, 0, 1], dtype=np.complex128)

KET_TLS_G = np.array([1, 0], dtype=np.complex128)
KET_TLS_E = np.array([0, 1], dtype=np.complex128)

PSI0 = np.kron(KET01, KET_TLS_G)
RHO0 = np.outer(PSI0, PSI0.conj())

# La phase relative est choisie conformément à l'évolution
# générée par exp(-i theta SX_SUB).
BELL_TARGET = np.kron(
    (KET01 - 1j * KET10) / np.sqrt(2),
    KET_TLS_G,
)

BELL_TARGET_4 = (
    KET01 - 1j * KET10
) / np.sqrt(2)

RHO0_4 = np.outer(KET01, KET01.conj())


# ============================================================
# VALIDATION DES OPÉRATEURS
# ============================================================

def hermiticity_error(matrix: ComplexMatrix) -> float:
    """Norme de Frobenius de H - H†."""
    return float(
        np.linalg.norm(matrix - matrix.conj().T, ord="fro")
    )


def validate_static_operators() -> None:
    """Échoue immédiatement si un opérateur statique est invalide."""

    operators = {
        "SX": SX,
        "SY": SY,
        "SZ": SZ,
        "SX_SUB": SX_SUB,
        "SY_SUB": SY_SUB,
        "SZ_SUB": SZ_SUB,
        "TX": TX,
        "TY": TY,
        "TZ": TZ,
    }

    for name, operator in operators.items():
        if not np.all(np.isfinite(operator)):
            raise ValueError(f"{name} contient des valeurs non finies.")

    for name in ("SX", "SY", "SZ", "SX_SUB", "SY_SUB", "SZ_SUB"):
        error = hermiticity_error(operators[name])
        if error > 1e-12:
            raise ValueError(
                f"{name} n'est pas hermitien : erreur={error:.3e}"
            )

    if not np.allclose(SX_SUB @ SX_SUB, np.diag([0, 1, 1, 0])):
        raise ValueError("Le carré de SX_SUB est incorrect.")

    log.info("Opérateurs statiques : validation réussie.")


# ============================================================
# FIDÉLITÉ ET TRACE PARTIELLE
# ============================================================

def partial_trace_tls(rho8: ComplexMatrix) -> ComplexMatrix:
    """Trace le TLS et conserve la matrice densité des deux qubits."""

    if rho8.shape != (8, 8):
        raise ValueError("La matrice d'entrée doit être de dimension 8x8.")

    # rho[i,a,j,b], où i,j indexent le système et a,b le TLS.
    tensor = rho8.reshape(4, 2, 4, 2)

    return np.trace(tensor, axis1=1, axis2=3)


def bell_fidelity(rho8: ComplexMatrix) -> float:
    """Fidélité avec l'état de Bell cible, après trace du TLS."""

    rho4 = partial_trace_tls(rho8)

    value = np.vdot(
        BELL_TARGET_4,
        rho4 @ BELL_TARGET_4,
    )

    return float(np.real(value))


def density_diagnostics(rho: ComplexMatrix) -> dict[str, float]:
    """Diagnostics numériques d'une matrice densité."""

    rho_h = 0.5 * (rho + rho.conj().T)

    return {
        "trace_real": float(np.real(np.trace(rho))),
        "trace_imag": float(np.imag(np.trace(rho))),
        "hermiticity_error": hermiticity_error(rho),
        "min_eigenvalue_hermitian_part": float(
            np.min(np.linalg.eigvalsh(rho_h))
        ),
    }


# ============================================================
# BRUITS CLASSIQUES
# ============================================================

def generate_1f_noise(
    n: int,
    dt: float,
    seed: int,
    f_min: float = 1e-2,
) -> np.ndarray:
    """
    Bruit coloré approchant un spectre en 1/f.

    Le mode DC est régularisé par f_min. La réalisation est centrée
    et normalisée. Le spectre est une approximation discrète.
    """

    if n < 4:
        raise ValueError("n doit être >= 4.")
    if dt <= 0 or f_min <= 0:
        raise ValueError("dt et f_min doivent être positifs.")

    rng = np.random.default_rng(seed)
    white = rng.normal(size=n)

    freqs = np.fft.rfftfreq(n, d=dt)
    safe_freqs = np.maximum(freqs, f_min)

    spectrum = np.fft.rfft(white) / np.sqrt(safe_freqs)
    noise = np.fft.irfft(spectrum, n=n)

    noise -= noise.mean()
    std = float(noise.std())

    if std < 1e-14:
        raise RuntimeError("Variance insuffisante pour le bruit 1/f.")

    return noise / std


def generate_rf_noise(
    t_eval: np.ndarray,
    freqs: tuple[float, ...],
    seed: int,
) -> np.ndarray:
    """Somme de composantes sinusoïdales de phases aléatoires."""

    rng = np.random.default_rng(seed + 10_000)
    signal = np.zeros_like(t_eval, dtype=float)

    for frequency in freqs:
        if frequency <= 0:
            raise ValueError("Les fréquences RF doivent être positives.")

        amplitude = rng.uniform(0.5, 1.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)

        signal += amplitude * np.cos(
            2.0 * np.pi * frequency * t_eval + phase
        )

    signal -= signal.mean()
    std = float(signal.std())

    if std < 1e-14:
        raise RuntimeError("Variance insuffisante pour le bruit RF.")

    return signal / std


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class WitnessConfig:
    t_span: tuple[float, float] = (0.0, 20.0)
    n_points: int = 501

    center: float = 10.0
    sigma: float = 2.0

    # Aire de pulse : pi/4 pour créer la cible de Bell idéale
    # lorsque le TLS est découplé et que le chirp est nul.
    exchange_fraction: float = 1.0

    n_mc: int = 24
    master_seed: int = 20261023

    noise_slow_level: float = 0.05
    noise_rf_level: float = 0.08
    rf_freqs: tuple[float, ...] = (0.5, 2.1, 5.0)

    # H_TLS = omega_tls/2 * TZ
    omega_tls: float = 0.3

    # Couplage système-TLS : nu_zx * kron(SZ_SUB, SX)
    nu_zx: float = 0.15

    # Taux de relaxation de population du TLS
    gamma_tls: float = 0.05

    # Taux de déphasage pur du TLS.
    # Le dissipateur associé à sqrt(gamma_phi/2)*TZ donne
    # gamma_phi * (TZ rho TZ - rho).
    gamma_phi_tls: float = 0.0

    kappas: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1)

    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = np.inf

    out_dir: Path = field(
        default_factory=lambda: Path("figures_tls")
    )


# ============================================================
# OPÉRATEURS DE LINDBLAD
# ============================================================

def build_lindblad_ops(cfg: WitnessConfig) -> list[ComplexMatrix]:
    """
    Relaxation et déphasage pur du TLS.

    Convention :
        L_relax = sqrt(gamma_tls) |g><e|
        L_phi   = sqrt(gamma_phi_tls / 2) * TZ

    Tous les taux doivent être >= 0.
    """

    if cfg.gamma_tls < 0 or cfg.gamma_phi_tls < 0:
        raise ValueError("Les taux de Lindblad doivent être >= 0.")

    sigma_minus = np.array(
        [[0, 1], [0, 0]],
        dtype=np.complex128,
    )

    operators: list[ComplexMatrix] = []

    if cfg.gamma_tls > 0:
        operators.append(
            np.sqrt(cfg.gamma_tls) * np.kron(I4, sigma_minus)
        )

    if cfg.gamma_phi_tls > 0:
        operators.append(
            np.sqrt(cfg.gamma_phi_tls / 2.0) * TZ
        )

    return operators


def lindblad_rhs(
    rho: ComplexMatrix,
    hamiltonian: ComplexMatrix,
    lindblad_ops: list[ComplexMatrix],
) -> ComplexMatrix:
    """Équation maîtresse de Lindblad."""

    drho = -1j * (
        hamiltonian @ rho - rho @ hamiltonian
    )

    for operator in lindblad_ops:
        ld_l = operator.conj().T @ operator

        drho += (
            operator @ rho @ operator.conj().T
            - 0.5 * (ld_l @ rho + rho @ ld_l)
        )

    return drho


# ============================================================
# INTÉGRATION DES MATRICES DE DENSITÉ
# ============================================================

def evolve_rho(
    rho0: ComplexMatrix,
    hamiltonian: Hamiltonian,
    lindblad_ops: list[ComplexMatrix],
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = np.inf,
) -> np.ndarray:
    """
    Intègre l'équation maîtresse avec DOP853.

    Le vecteur complexe est représenté par la concaténation
    de ses parties réelle et imaginaire. Aucune projection
    artificielle sur l'espace des matrices densité n'est appliquée.
    """

    if rho0.ndim != 2 or rho0.shape[0] != rho0.shape[1]:
        raise ValueError("rho0 doit être une matrice carrée.")

    dim = rho0.shape[0]

    if not np.allclose(rho0, rho0.conj().T, atol=1e-10):
        raise ValueError("rho0 doit être hermitienne.")

    if abs(np.trace(rho0) - 1.0) > 1e-10:
        raise ValueError("La trace initiale de rho0 doit être 1.")

    if np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval doit être strictement croissant.")

    if not np.allclose(
        [t_eval[0], t_eval[-1]],
        list(t_span),
        atol=1e-10,
    ):
        raise ValueError("t_eval doit couvrir entièrement t_span.")

    if max_step <= 0:
        raise ValueError("max_step doit être positif.")

    def pack(matrix: ComplexMatrix) -> np.ndarray:
        return np.concatenate([
            matrix.real.ravel(),
            matrix.imag.ravel(),
        ])

    def unpack(vector: np.ndarray) -> ComplexMatrix:
        size = dim * dim
        return (
            vector[:size] + 1j * vector[size:]
        ).reshape(dim, dim)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        rho = unpack(y)
        H = hamiltonian(t)

        if H.shape != (dim, dim):
            raise ValueError("Dimension incorrecte du Hamiltonien.")

        drho = lindblad_rhs(rho, H, lindblad_ops)
        return pack(drho)

    solution = solve_ivp(
        rhs,
        t_span,
        pack(rho0),
        t_eval=t_eval,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    if not solution.success:
        raise RuntimeError(
            f"Échec intégration : {solution.message}"
        )

    size = dim * dim

    states = np.array([
        (
            column[:size] + 1j * column[size:]
        ).reshape(dim, dim)
        for column in solution.y.T
    ])

    if not np.all(np.isfinite(states)):
        raise FloatingPointError(
            "La solution contient des valeurs non finies."
        )

    return states


# ============================================================
# CONSTRUCTION DES PROTOCOLES
# ============================================================

def build_experiment(
    cfg: WitnessConfig,
    seed: int,
):
    """
    Construit trois protocoles utilisant les mêmes réalisations
    de bruit :

    forward:
        Commande et bruit dans leur ordre direct.

    reversed_both:
        Commande et bruit classique parcourus en ordre inversé.
        La dissipation de Lindblad reste inchangée.

    reversed_control:
        Commande inversée, bruit classique conservé dans l'ordre
        direct. Ce contraste teste la sensibilité au profil temporel
        du bruit sous commande inversée.
    """

    t_eval = np.linspace(
        cfg.t_span[0],
        cfg.t_span[1],
        cfg.n_points,
    )

    dt = float(t_eval[1] - t_eval[0])
    t_start, t_end = cfg.t_span

    if cfg.sigma <= 0:
        raise ValueError("sigma doit être strictement positif.")

    if not (t_start <= cfg.center <= t_end):
        raise ValueError("center doit appartenir à t_span.")

    if cfg.noise_slow_level < 0 or cfg.noise_rf_level < 0:
        raise ValueError("Les niveaux de bruit doivent être >= 0.")

    env = np.exp(
        -0.5 * ((t_eval - cfg.center) / cfg.sigma) ** 2
    )

    envelope_area = float(np.trapezoid(env, t_eval))

    if envelope_area <= 0:
        raise RuntimeError("Aire de l'enveloppe invalide.")

    # Pour kappa = 0, nu_zx = 0 et sans dissipation/bruit,
    # l'intégrale du couplage SX_SUB vaut pi/4 * exchange_fraction.
    j_peak = (
        (np.pi / 4.0)
        * cfg.exchange_fraction
        / envelope_area
    )

    slow = generate_1f_noise(
        cfg.n_points,
        dt,
        seed,
    )

    rf = generate_rf_noise(
        t_eval,
        cfg.rf_freqs,
        seed,
    )

    def noise_delta(t: float) -> float:
        slow_value = np.interp(t, t_eval, slow)
        rf_value = np.interp(t, t_eval, rf)

        return (
            cfg.noise_slow_level * slow_value
            + cfg.noise_rf_level * rf_value
        )

    def control(t: float, kappa: float) -> ComplexMatrix:
        envelope = float(np.interp(t, t_eval, env))
        phase = kappa * (t - cfg.center)

        h_exchange = j_peak * envelope * (
            np.cos(phase) * SX_SUB
            + np.sin(phase) * SY_SUB
        )

        return np.kron(h_exchange, I2)

    # Bruit longitudinal appliqué au système à deux qubits.
    def noise(t: float) -> ComplexMatrix:
        h_noise = 0.5 * noise_delta(t) * SZ_SUB
        return np.kron(h_noise, I2)

    # Hamiltonien statique du TLS et de son couplage.
    H_tls = (
        0.5 * cfg.omega_tls * TZ
        + cfg.nu_zx * np.kron(SZ_SUB, SX)
    )

    def h_forward(kappa: float) -> Hamiltonian:
        def H(t: float) -> ComplexMatrix:
            return (
                control(t, kappa)
                + noise(t)
                + H_tls
            )
        return H

    def h_reversed_both(kappa: float) -> Hamiltonian:
        def H(t: float) -> ComplexMatrix:
            tr = t_start + t_end - t
            return (
                control(tr, kappa)
                + noise(tr)
                + H_tls
            )
        return H

    def h_reversed_control(kappa: float) -> Hamiltonian:
        def H(t: float) -> ComplexMatrix:
            tr = t_start + t_end - t
            return (
                control(tr, kappa)
                + noise(t)
                + H_tls
            )
        return H

    return (
        t_eval,
        h_forward,
        h_reversed_both,
        h_reversed_control,
        build_lindblad_ops(cfg),
        j_peak,
    )


# ============================================================
# CONTRÔLE DE RÉVERSIBILITÉ UNITAIRE
# ============================================================

def unitary_round_trip_error(
    cfg: WitnessConfig,
    t_eval: np.ndarray,
    hamiltonian: Hamiltonian,
) -> float:
    """
    Contrôle indépendant, sans dissipation.

    1. Évolue de rho0 jusqu'à T avec H(t).
    2. Repart de rho(T) avec H_back(s) = -H(T-s).
    3. Compare la matrice récupérée à rho0.

    Ce test vérifie une inversion dynamique construite
    mathématiquement, contrairement à une simple inversion
    du profil de commande.
    """

    forward = evolve_rho(
        RHO0,
        hamiltonian,
        [],
        cfg.t_span,
        t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step,
    )

    rho_final = forward[-1]
    t_start, t_end = cfg.t_span

    def h_backward(s: float) -> ComplexMatrix:
        return -hamiltonian(t_start + t_end - s)

    backward = evolve_rho(
        rho_final,
        h_backward,
        [],
        cfg.t_span,
        t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step,
    )

    return float(
        np.linalg.norm(
            backward[-1] - RHO0,
            ord="fro",
        )
    )


# ============================================================
# VALIDATION PHYSIQUE D'UNE TRAJECTOIRE
# ============================================================

def validate_trajectory(
    states: np.ndarray,
    trace_tolerance: float = 2e-6,
    hermiticity_tolerance: float = 2e-6,
    positivity_tolerance: float = 2e-6,
) -> dict[str, float | bool]:
    """
    Mesure les écarts numériques sur toute la trajectoire.

    La positivité est contrôlée via les valeurs propres de la
    partie hermitienne. Les seuils servent au diagnostic :
    aucune correction silencieuse des états n'est appliquée.
    """

    trace_errors = []
    hermiticity_errors = []
    minimum_eigenvalues = []

    for rho in states:
        trace_errors.append(
            abs(np.trace(rho) - 1.0)
        )

        hermiticity_errors.append(
            hermiticity_error(rho)
        )

        rho_h = 0.5 * (rho + rho.conj().T)

        minimum_eigenvalues.append(
            float(np.min(np.linalg.eigvalsh(rho_h)))
        )

    max_trace_error = float(np.max(trace_errors))
    max_hermiticity_error = float(
        np.max(hermiticity_errors)
    )
    min_eigenvalue = float(
        np.min(minimum_eigenvalues)
    )

    valid = (
        max_trace_error <= trace_tolerance
        and max_hermiticity_error <= hermiticity_tolerance
        and min_eigenvalue >= -positivity_tolerance
    )

    return {
        "max_trace_error": max_trace_error,
        "max_hermiticity_error": max_hermiticity_error,
        "min_eigenvalue": min_eigenvalue,
        "valid_within_tolerances": bool(valid),
    }


# ============================================================
# MONTE-CARLO APPARIÉ
# ============================================================

def compute_A(
    kappa: float,
    cfg: WitnessConfig,
    seeds=None,
) -> dict:
    """
    Compare les trois protocoles pour les mêmes graines.

    A_control_reversed:
        F_forward - F_reversed_control

    A_both_reversed:
        F_forward - F_reversed_both

    Les écarts sont appariés par graine.
    """

    if seeds is None:
        seeds = range(
            cfg.master_seed,
            cfg.master_seed + cfg.n_mc,
        )

    seeds = list(seeds)

    if len(seeds) < 1:
        raise ValueError("Au moins une graine est requise.")

    t_eval = np.linspace(
        cfg.t_span[0],
        cfg.t_span[1],
        cfg.n_points,
    )

    fidelity_forward = []
    fidelity_both_reversed = []
    fidelity_control_reversed = []

    paired_both = []
    paired_control = []

    diagnostics = []
    round_trip_errors = []

    for index, seed in enumerate(seeds):
        (
            t,
            h_f,
            h_rb,
            h_rc,
            Lops,
            _j_peak,
        ) = build_experiment(cfg, int(seed))

        # Trajectoire directe.
        states_f = evolve_rho(
            RHO0,
            h_f(kappa),
            Lops,
            cfg.t_span,
            t,
            rtol=cfg.rtol,
            atol=cfg.atol,
            max_step=cfg.max_step,
        )

        # Commande et bruit parcourus à rebours.
        states_rb = evolve_rho(
            RHO0,
            h_rb(kappa),
            Lops,
            cfg.t_span,
            t,
            rtol=cfg.rtol,
            atol=cfg.atol,
            max_step=cfg.max_step,
        )

        # Commande inversée, bruit dans l'ordre direct.
        states_rc = evolve_rho(
            RHO0,
            h_rc(kappa),
            Lops,
            cfg.t_span,
            t,
            rtol=cfg.rtol,
            atol=cfg.atol,
            max_step=cfg.max_step,
        )

        ff = bell_fidelity(states_f[-1])
        frb = bell_fidelity(states_rb[-1])
        frc = bell_fidelity(states_rc[-1])

        fidelity_forward.append(ff)
        fidelity_both_reversed.append(frb)
        fidelity_control_reversed.append(frc)

        paired_both.append(ff - frb)
        paired_control.append(ff - frc)

        if index == 0:
            diagnostics.append({
                "forward": validate_trajectory(states_f),
                "reversed_both": validate_trajectory(states_rb),
                "reversed_control": validate_trajectory(states_rc),
            })

            # Contrôle unitaire indépendant de la dissipation.
            round_trip_errors.append(
                unitary_round_trip_error(
                    cfg,
                    t,
                    h_f(kappa),
                )
            )

    paired_both_array = np.asarray(paired_both)
    paired_control_array = np.asarray(paired_control)

    def summary(values: np.ndarray) -> dict:
        values = np.asarray(values, dtype=float)

        return {
            "mean": float(np.mean(values)),
            "std_sample": float(
                np.std(values, ddof=1)
                if values.size > 1 else 0.0
            ),
            "sem": float(
                np.std(values, ddof=1) / np.sqrt(values.size)
                if values.size > 1 else 0.0
            ),
            "abs_max": float(np.max(np.abs(values))),
            "n": int(values.size),
        }

    return {
        "kappa": float(kappa),
        "n_mc": len(seeds),
        "seeds": [int(seed) for seed in seeds],

        "F_forward": summary(fidelity_forward),
        "F_reversed_both": summary(fidelity_both_reversed),
        "F_reversed_control": summary(fidelity_control_reversed),

        "A_both_reversed": summary(paired_both_array),
        "A_control_reversed": summary(paired_control_array),

        "unitary_round_trip_error": (
            float(round_trip_errors[0])
            if round_trip_errors else None
        ),

        "trajectory_diagnostics_first_seed": (
            diagnostics[0] if diagnostics else None
        ),
    }


# ============================================================
# FIGURE
# ============================================================

def plot_witness(
    cfg: WitnessConfig,
    results: dict,
) -> Path:
    kappas = np.array([
        float(k) for k in results.keys()
    ])

    means_control = np.array([
        results[k]["A_control_reversed"]["mean"]
        for k in results
    ])

    errors_control = np.array([
        results[k]["A_control_reversed"]["sem"]
        for k in results
    ])

    means_both = np.array([
        results[k]["A_both_reversed"]["mean"]
        for k in results
    ])

    errors_both = np.array([
        results[k]["A_both_reversed"]["sem"]
        for k in results
    ])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.errorbar(
        kappas,
        means_control,
        yerr=errors_control,
        marker="o",
        capsize=4,
        label="Commande inversée, bruit direct",
    )

    ax.errorbar(
        kappas,
        means_both,
        yerr=errors_both,
        marker="s",
        capsize=4,
        label="Commande et bruit inversés",
    )

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel(r"Chirp $\kappa$")
    ax.set_ylabel(r"$\Delta F_{\mathrm{Bell}}$")
    ax.set_title("Témoin d'ordre temporel — système qubit + TLS")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    output = cfg.out_dir / "temoin_inversion_tls.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    return output


# ============================================================
# RAPPORT JSON
# ============================================================

def json_safe(value):
    """Convertit les types NumPy et Path en valeurs JSON standard."""

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    return value


# ============================================================
# EXÉCUTION PRINCIPALE
# ============================================================

def run_witness(cfg: WitnessConfig | None = None) -> dict:
    if cfg is None:
        cfg = WitnessConfig()

    if cfg.n_points < 3:
        raise ValueError("n_points doit être >= 3.")

    if cfg.n_mc < 1:
        raise ValueError("n_mc doit être >= 1.")

    if cfg.t_span[1] <= cfg.t_span[0]:
        raise ValueError("t_span doit être croissant.")

    if cfg.exchange_fraction < 0:
        raise ValueError("exchange_fraction doit être >= 0.")

    validate_static_operators()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 68)
    log.info("AXIOMES BASE LABS — TIME-REVERSAL NOISE WITNESS")
    log.info("Système : deux qubits + TLS, dimension 8")
    log.info("Équation : Lindblad avec bruit classique")
    log.info("Nombre de réalisations par kappa : %d", cfg.n_mc)
    log.info("=" * 68)

    results = {}

    for kappa in cfg.kappas:
        log.info("Calcul pour kappa = %.6g", kappa)

        stat = compute_A(kappa, cfg)
        key = f"{kappa:.6g}"
        results[key] = stat

        a_control = stat["A_control_reversed"]
        a_both = stat["A_both_reversed"]

        log.info(
            "  A_control = %.4e +/- %.4e (SEM)",
            a_control["mean"],
            a_control["sem"],
        )

        log.info(
            "  A_both    = %.4e +/- %.4e (SEM)",
            a_both["mean"],
            a_both["sem"],
        )

        log.info(
            "  Erreur round-trip unitaire = %.4e",
            stat["unitary_round_trip_error"],
        )

    figure_path = plot_witness(cfg, results)

    report = {
        "metadata": {
            "module": "AXIOMES BASE LABS — Time-Reversal Noise Witness",
            "version": "2.0.0",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "hbar": 1.0,
            "hilbert_dimension": 8,
            "system_dimension": 4,
            "tls_dimension": 2,
            "interpretation": (
                "Les asymétries mesurent des différences de fidélité "
                "entre protocoles simulés. Elles ne prouvent pas seules "
                "une irréversibilité fondamentale ni une "
                "non-markovianité."
            ),
        },
        "configuration": json_safe(asdict(cfg)),
        "results": results,
        "figure": str(figure_path.resolve()),
    }

    report_path = cfg.out_dir / "rapport_temoin_tls.json"

    report_path.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    log.info("-" * 68)
    log.info("Rapport : %s", report_path.resolve())
    log.info("Figure  : %s", figure_path.resolve())
    log.info("Calcul terminé.")
    log.info(
        "Les diagnostics de trace, d'hermiticité et de positivité "
        "sont consignés dans le rapport."
    )

    return report


if __name__ == "__main__":
    run_witness(
        WitnessConfig(
            n_mc=24,
            n_points=501,
        )
  )
