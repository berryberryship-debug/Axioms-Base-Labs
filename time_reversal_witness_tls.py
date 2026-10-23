#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXIOMES BASE LABS — Time-Reversal Noise Witness
Modèle Qubit + TLS non-Markovien (Matrices de densité & Lindblad)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ============================================================
# Opérateurs de base (2 qubits + 1 TLS → dim = 8)
# ============================================================
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

SX_SUB = np.kron(SX, I2) @ np.array([
    [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]
], dtype=complex)
SY_SUB = np.array([
    [0, 0, 0, 0], [0, 0, -1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]
], dtype=complex)
SZ_SUB = np.array([
    [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]
], dtype=complex)
I4 = np.eye(4, dtype=complex)

TX = np.kron(I4, SX)
TY = np.kron(I4, SY)
TZ = np.kron(I4, SZ)
I8 = np.eye(8, dtype=complex)

KET01 = np.array([0, 1, 0, 0], dtype=complex)
KET_TLS_G = np.array([1, 0], dtype=complex)
PSI0 = np.kron(KET01, KET_TLS_G)

BELL_PSI_PLUS = (np.array([0, 1, 0, 0]) + np.array([0, 0, 1, 0])) / np.sqrt(2)

def reduced_bell_fidelity(rho8: np.ndarray) -> float:
    rho4 = np.trace(rho8.reshape(4, 2, 4, 2), axis1=1, axis2=3)
    return float(np.real(np.vdot(BELL_PSI_PLUS, rho4 @ BELL_PSI_PLUS)))

# ============================================================
# Générateurs de bruit
# ============================================================
def generate_1f_noise(n: int, dt: float, seed: int, f_min: float = 1e-2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(size=n)
    freqs = np.fft.rfftfreq(n, d=dt)
    freqs[0] = f_min
    spectrum_ft = np.fft.rfft(white) / np.sqrt(freqs)
    noise = np.fft.irfft(spectrum_ft, n=n)
    noise -= noise.mean()
    return noise / (noise.std() + 1e-15)

def generate_rf_noise(t_eval: np.ndarray, freqs: tuple[float, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_000)
    signal = np.zeros_like(t_eval, dtype=float)
    for f in freqs:
        amplitude = rng.uniform(0.5, 1.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += amplitude * np.cos(2.0 * np.pi * f * t_eval + phase)
    signal -= signal.mean()
    return signal / (signal.std() + 1e-15)

# ============================================================
# Évolution de la matrice densité (Lindblad)
# ============================================================
def evolve_rho(rho0: np.ndarray, hamiltonian: Callable[[float], np.ndarray], lindblad_ops: list[np.ndarray], t_span: tuple[float, float], t_eval: np.ndarray) -> np.ndarray:
    dim = rho0.shape[0]
    y0 = np.concatenate([rho0.real.ravel(), rho0.imag.ravel()])

    def rhs(t, y):
        rho = (y[:dim*dim] + 1j * y[dim*dim:]).reshape(dim, dim)
        H = hamiltonian(t)
        drho = -1j * (H @ rho - rho @ H)
        for L in lindblad_ops:
            drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        return np.concatenate([drho.real.ravel(), drho.imag.ravel()])

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="DOP853", rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"Échec intégration : {sol.message}")
    return np.array([(col[:dim*dim] + 1j * col[dim*dim:]).reshape(dim, dim) for col in sol.y.T])

@dataclass(frozen=True)
class WitnessConfig:
    t_span: tuple[float, float] = (0.0, 20.0)
    n_points: int = 500
    center: float = 10.0
    sigma: float = 2.0
    exchange_fraction: float = 0.6
    n_mc: int = 4
    noise_slow_level: float = 0.05
    noise_rf_level: float = 0.08
    kappas: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1)
    rf_freqs: tuple[float, ...] = (0.5, 2.1, 5.0)
    nu_zx: float = 0.15
    omega_tls: float = 0.3
    kappa_tls: float = 0.05
    out_dir: Path = field(default_factory=lambda: Path("figures_tls"))

def build_experiment(cfg: WitnessConfig, seed: int):
    t_eval = np.linspace(*cfg.t_span, cfg.n_points)
    dt = float(t_eval[1] - t_eval[0])
    t_end = float(cfg.t_span[1])
    env = np.exp(-0.5 * ((t_eval - cfg.center) / cfg.sigma) ** 2)
    j_peak = np.pi / float(np.trapezoid(env, t_eval)) * cfg.exchange_fraction
    
    slow = generate_1f_noise(cfg.n_points, dt, seed)
    rf = generate_rf_noise(t_eval, cfg.rf_freqs, seed)

    def noise_delta(t): return cfg.noise_slow_level * np.interp(t, t_eval, slow) + cfg.noise_rf_level * np.interp(t, t_eval, rf)
    def control(t, kappa): return np.kron(j_peak * np.interp(t, t_eval, env) * (np.cos(kappa * (t - cfg.center)) * SX_SUB + np.sin(kappa * (t - cfg.center)) * SY_SUB), I2)
    def noise(t): return np.kron(0.5 * noise_delta(t) * SZ_SUB, I2)
    def H_tls(): return 0.5 * cfg.omega_tls * TZ + cfg.nu_zx * np.kron(SZ_SUB, SX)

    L_tls = np.sqrt(cfg.kappa_tls) * np.kron(I4, np.array([[0, 1], [0, 0]], dtype=complex))

    return t_eval, lambda k: lambda t: control(t, k) + noise(t) + H_tls(), lambda k: lambda t: control(t_end - t, k) + noise(t_end - t) + H_tls(), lambda k: lambda t: control(t_end - t, k) + noise(t) + H_tls(), [L_tls]

def compute_A(kappa: float, cfg: WitnessConfig, seeds=None) -> dict:
    if seeds is None: seeds = range(cfg.n_mc)
    a_exact, a_phys, f_forward, f_reversed = [], [], [], []
    t_eval = np.linspace(*cfg.t_span, cfg.n_points)
    rho0 = np.outer(PSI0, PSI0.conj())
    
    for seed in seeds:
        _, h_f, h_er, h_pr, Lops = build_experiment(cfg, seed)
        rho_f = evolve_rho(rho0, h_f(kappa), Lops, cfg.t_span, t_eval)[-1]
        rho_e = evolve_rho(rho0, h_er(kappa), Lops, cfg.t_span, t_eval)[-1]
        rho_p = evolve_rho(rho0, h_pr(kappa), Lops, cfg.t_span, t_eval)[-1]
        
        f0 = reduced_bell_fidelity(rho_f)
        f_e = reduced_bell_fidelity(rho_e)
        f_p = reduced_bell_fidelity(rho_p)
        
        f_forward.append(f0); f_reversed.append(f_p)
        a_exact.append(f0 - f_e); a_phys.append(f0 - f_p)
        
    a_exact, a_phys = np.array(a_exact), np.array(a_phys)
    return {
        "kappa": float(kappa),
        "A_exact_abs_max": float(np.max(np.abs(a_exact))),
        "A_phys_mean": float(np.mean(a_phys)),
        "A_phys_std": float(np.std(a_phys, ddof=1) if len(a_phys) > 1 else 0.0),
        "A_phys_abs_max": float(np.max(np.abs(a_phys))),
        "F_forward_mean": float(np.mean(f_forward)),
        "F_reversed_mean": float(np.mean(f_reversed)),
    }

def run_witness(cfg: WitnessConfig = WitnessConfig()):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    print("=" * 60)
    print("TÉMOIN D'INVERSION TEMPORELLE + MODÈLE QUBIT-TLS")
    print("=" * 60)

    for kappa in cfg.kappas:
        stat = compute_A(kappa, cfg)
        results[f"{kappa:.6g}"] = stat
        print(f"kappa={kappa:6.3f} | A_exact={stat['A_exact_abs_max']:.2e} | A_phys={stat['A_phys_mean']:.3e} ± {stat['A_phys_std']:.3e}")

    (cfg.out_dir / "rapport_temoin_tls.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSUCCÈS : rapport sauvegardé dans {cfg.out_dir.resolve()}")

if __name__ == "__main__":
    run_witness(WitnessConfig(n_mc=4, n_points=500))
