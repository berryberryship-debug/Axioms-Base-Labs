#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXIOMES BASE LABS
Time-Reversal Noise Witness — Qubit + TLS
Version: 2.0.0
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("AXIOMES.TimeReversal")

ComplexMatrix = np.ndarray
Hamiltonian = Callable[[float], ComplexMatrix]

SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)

I2 = np.eye(2, dtype=np.complex128)
I4 = np.eye(4, dtype=np.complex128)
I8 = np.eye(8, dtype=np.complex128)

SX_SUB = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.complex128)
SY_SUB = np.array([[0, 0, 0, 0], [0, 0, -1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]], dtype=np.complex128)
SZ_SUB = np.diag([0, 1, -1, 0]).astype(np.complex128)

TX = np.kron(I4, SX)
TY = np.kron(I4, SY)
TZ = np.kron(I4, SZ)

KET00 = np.array([1, 0, 0, 0], dtype=np.complex128)
KET01 = np.array([0, 1, 0, 0], dtype=np.complex128)
KET10 = np.array([0, 0, 1, 0], dtype=np.complex128)
KET11 = np.array([0, 0, 0, 1], dtype=np.complex128)

KET_TLS_G = np.array([1, 0], dtype=np.complex128)
KET_TLS_E = np.array([0, 1], dtype=np.complex128)

PSI0 = np.kron(KET01, KET_TLS_G)
RHO0 = np.outer(PSI0, PSI0.conj())
BELL_TARGET_4 = (KET01 - 1j * KET10) / np.sqrt(2)

def hermiticity_error(matrix: ComplexMatrix) -> float:
    return float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))

def validate_static_operators() -> None:
    operators = {"SX": SX, "SY": SY, "SZ": SZ, "SX_SUB": SX_SUB, "SY_SUB": SY_SUB, "SZ_SUB": SZ_SUB, "TX": TX, "TY": TY, "TZ": TZ}
    for name, operator in operators.items():
        if not np.all(np.isfinite(operator)):
            raise ValueError(f"{name} contient des valeurs non finies.")
    for name in ("SX", "SY", "SZ", "SX_SUB", "SY_SUB", "SZ_SUB"):
        if hermiticity_error(operators[name]) > 1e-12:
            raise ValueError(f"{name} n'est pas hermitien.")
    if not np.allclose(SX_SUB @ SX_SUB, np.diag([0, 1, 1, 0])):
        raise ValueError("Le carré de SX_SUB est incorrect.")
    log.info("Opérateurs statiques : validation réussie.")

def partial_trace_tls(rho8: ComplexMatrix) -> ComplexMatrix:
    return np.trace(rho8.reshape(4, 2, 4, 2), axis1=1, axis2=3)

def bell_fidelity(rho8: ComplexMatrix) -> float:
    rho4 = partial_trace_tls(rho8)
    return float(np.real(np.vdot(BELL_TARGET_4, rho4 @ BELL_TARGET_4)))

def generate_1f_noise(n: int, dt: float, seed: int, f_min: float = 1e-2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(size=n)
    freqs = np.fft.rfftfreq(n, d=dt)
    safe_freqs = np.maximum(freqs, f_min)
    spectrum = np.fft.rfft(white) / np.sqrt(safe_freqs)
    noise = np.fft.irfft(spectrum, n=n)
    noise -= noise.mean()
    return noise / (noise.std() + 1e-15)

def generate_rf_noise(t_eval: np.ndarray, freqs: tuple[float, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_000)
    signal = np.zeros_like(t_eval, dtype=float)
    for f in freqs:
        signal += rng.uniform(0.5, 1.0) * np.cos(2.0 * np.pi * f * t_eval + rng.uniform(0.0, 2.0 * np.pi))
    signal -= signal.mean()
    return signal / (signal.std() + 1e-15)

@dataclass(frozen=True)
class WitnessConfig:
    t_span: tuple[float, float] = (0.0, 20.0)
    n_points: int = 501
    center: float = 10.0
    sigma: float = 2.0
    exchange_fraction: float = 1.0
    n_mc: int = 24
    master_seed: int = 20261023
    noise_slow_level: float = 0.05
    noise_rf_level: float = 0.08
    rf_freqs: tuple[float, ...] = (0.5, 2.1, 5.0)
    omega_tls: float = 0.3
    nu_zx: float = 0.15
    gamma_tls: float = 0.05
    gamma_phi_tls: float = 0.0
    kappas: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1)
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = np.inf
    out_dir: Path = field(default_factory=lambda: Path("figures_tls"))

def build_lindblad_ops(cfg: WitnessConfig) -> list[ComplexMatrix]:
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    operators = []
    if cfg.gamma_tls > 0:
        operators.append(np.sqrt(cfg.gamma_tls) * np.kron(I4, sigma_minus))
    if cfg.gamma_phi_tls > 0:
        operators.append(np.sqrt(cfg.gamma_phi_tls / 2.0) * TZ)
    return operators

def lindblad_rhs(rho: ComplexMatrix, hamiltonian: ComplexMatrix, lindblad_ops: list[ComplexMatrix]) -> ComplexMatrix:
    drho = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    for L in lindblad_ops:
        ld_l = L.conj().T @ L
        drho += (L @ rho @ L.conj().T - 0.5 * (ld_l @ rho + rho @ ld_l))
    return drho

def evolve_rho(rho0: ComplexMatrix, hamiltonian: Hamiltonian, lindblad_ops: list[ComplexMatrix], t_span: tuple[float, float], t_eval: np.ndarray, *, rtol: float = 1e-8, atol: float = 1e-10, max_step: float = np.inf) -> np.ndarray:
    dim = rho0.shape[0]
    def pack(m): return np.concatenate([m.real.ravel(), m.imag.ravel()])
    def unpack(v): return (v[:dim*dim] + 1j * v[dim*dim:]).reshape(dim, dim)
    def rhs(t, y):
        rho = unpack(y)
        H = hamiltonian(t)
        drho = lindblad_rhs(rho, H, lindblad_ops)
        return pack(drho)
    sol = solve_ivp(rhs, t_span, pack(rho0), t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success:
        raise RuntimeError(f"Échec intégration : {sol.message}")
    size = dim * dim
    return np.array([(col[:size] + 1j * col[size:]).reshape(dim, dim) for col in sol.y.T])

def build_experiment(cfg: WitnessConfig, seed: int):
    t_eval = np.linspace(cfg.t_span[0], cfg.t_span[1], cfg.n_points)
    dt = float(t_eval[1] - t_eval[0])
    t_start, t_end = cfg.t_span
    env = np.exp(-0.5 * ((t_eval - cfg.center) / cfg.sigma) ** 2)
    envelope_area = float(np.trapezoid(env, t_eval))
    j_peak = ((np.pi / 4.0) * cfg.exchange_fraction / envelope_area)
    slow = generate_1f_noise(cfg.n_points, dt, seed)
    rf = generate_rf_noise(t_eval, cfg.rf_freqs, seed)

    def noise_delta(t): return cfg.noise_slow_level * np.interp(t, t_eval, slow) + cfg.noise_rf_level * np.interp(t, t_eval, rf)
    def control(t, kappa):
        phase = kappa * (t - cfg.center)
        h_ex = j_peak * np.interp(t, t_eval, env) * (np.cos(phase) * SX_SUB + np.sin(phase) * SY_SUB)
        return np.kron(h_ex, I2)
    def noise(t): return np.kron(0.5 * noise_delta(t) * SZ_SUB, I2)
    H_tls = (0.5 * cfg.omega_tls * TZ + cfg.nu_zx * np.kron(SZ_SUB, SX))

    return (
        t_eval,
        lambda k: lambda t: control(t, k) + noise(t) + H_tls,
        lambda k: lambda t: control(t_start + t_end - t, k) + noise(t_start + t_end - t) + H_tls,
        lambda k: lambda t: control(t_start + t_end - t, k) + noise(t) + H_tls,
        build_lindblad_ops(cfg)
    )

def unitary_round_trip_error(cfg: WitnessConfig, t_eval: np.ndarray, hamiltonian: Hamiltonian) -> float:
    forward = evolve_rho(RHO0, hamiltonian, [], cfg.t_span, t_eval)
    t_start, t_end = cfg.t_span
    backward = evolve_rho(forward[-1], lambda s: -hamiltonian(t_start + t_end - s), [], cfg.t_span, t_eval)
    return float(np.linalg.norm(backward[-1] - RHO0, ord="fro"))

def compute_A(kappa: float, cfg: WitnessConfig) -> dict:
    seeds = range(cfg.master_seed, cfg.master_seed + cfg.n_mc)
    t_eval = np.linspace(cfg.t_span[0], cfg.t_span[1], cfg.n_points)
    paired_both, paired_control, round_trip = [], [], []
    for i, seed in enumerate(seeds):
        t, h_f, h_rb, h_rc, Lops = build_experiment(cfg, int(seed))
        f = bell_fidelity(evolve_rho(RHO0, h_f(kappa), Lops, cfg.t_span, t)[-1])
        rb = bell_fidelity(evolve_rho(RHO0, h_rb(kappa), Lops, cfg.t_span, t)[-1])
        rc = bell_fidelity(evolve_rho(RHO0, h_rc(kappa), Lops, cfg.t_span, t)[-1])
        paired_both.append(f - rb)
        paired_control.append(f - rc)
        if i == 0: round_trip.append(unitary_round_trip_error(cfg, t, h_f(kappa)))
    def summary(v): return {"mean": float(np.mean(v)), "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0}
    return {"kappa": float(kappa), "A_both": summary(paired_both), "A_control": summary(paired_control), "round_trip": round_trip[0]}

def run_witness(cfg: WitnessConfig):
    validate_static_operators()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print("AXIOMES BASE LABS — TIME-REVERSAL NOISE WITNESS")
    print("=" * 68)
    results = {}
    for kappa in cfg.kappas:
        print(f"Calcul pour kappa = {kappa}...", end=" ", flush=True)
        stat = compute_A(kappa, cfg)
        results[str(kappa)] = stat
        print(f"OK\n  A_control = {stat['A_control']['mean']:+.4e} +/- {stat['A_control']['sem']:.4e} (SEM)\n  A_both    = {stat['A_both']['mean']:+.4e} +/- {stat['A_both']['sem']:.4e} (SEM)\n  Erreur round-trip = {stat['round_trip']:.4e}")
    print("Calcul terminé.")

if __name__ == "__main__":
    run_witness(WitnessConfig(n_mc=24, n_points=501))
