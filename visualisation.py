#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXIOMES BASE LABS — Time-Reversal Noise Witness
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

# ==========================================
# OPÉRATEURS ET ÉTATS
# ==========================================
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

KET01 = np.array([0, 1, 0, 0], dtype=complex)
KET10 = np.array([0, 0, 1, 0], dtype=complex)

SX_SUB = np.kron(SX, I2) @ np.array([
    [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]
], dtype=complex)

SY_SUB = np.array([
    [0, 0, 0, 0], [0, 0, -1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]
], dtype=complex)

SZ_SUB = np.array([
    [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]
], dtype=complex)

BELL_PSI_PLUS = (KET01 + KET10) / np.sqrt(2)

def bell_fidelity(psi: np.ndarray) -> float:
    return float(abs(np.vdot(BELL_PSI_PLUS, psi)) ** 2)

# ==========================================
# GÉNÉRATEURS DE BRUIT
# ==========================================
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

# ==========================================
# ÉVOLUTION QUANTIQUE
# ==========================================
def evolve(psi0: np.ndarray, hamiltonian: Callable[[float], np.ndarray], t_span: tuple[float, float], t_eval: np.ndarray) -> np.ndarray:
    dim = psi0.size
    def rhs(_t, y):
        psi = y[:dim] + 1j * y[dim:]
        dpsi = -1j * hamiltonian(_t) @ psi
        return np.concatenate([dpsi.real, dpsi.imag])
    y0 = np.concatenate([psi0.real, psi0.imag])
    solution = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="DOP853", rtol=1e-10, atol=1e-12)
    if not solution.success:
        raise RuntimeError(f"Échec de l'intégration : {solution.message}")
    return (solution.y[:dim] + 1j * solution.y[dim:]).T

# ==========================================
# EXPÉRIENCE
# ==========================================
@dataclass(frozen=True)
class WitnessConfig:
    t_span: tuple[float, float] = (0.0, 20.0)
    n_points: int = 1200
    center: float = 10.0
    sigma: float = 2.0
    exchange_fraction: float = 0.6
    n_mc: int = 24
    noise_slow_level: float = 0.05
    noise_rf_level: float = 0.08
    kappas: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1)
    rf_freqs: tuple[float, ...] = (0.5, 2.1, 5.0)
    out_dir: Path = field(default_factory=lambda: Path("figures"))

def build_experiment(cfg: WitnessConfig, seed: int):
    t_eval = np.linspace(*cfg.t_span, cfg.n_points)
    dt = float(t_eval[1] - t_eval[0])
    t_end = float(cfg.t_span[1])
    env = np.exp(-0.5 * ((t_eval - cfg.center) / cfg.sigma) ** 2)
    j_peak = np.pi / float(np.trapezoid(env, t_eval)) * cfg.exchange_fraction
    slow = generate_1f_noise(cfg.n_points, dt, seed)
    rf = generate_rf_noise(t_eval, cfg.rf_freqs, seed)

    def noise_delta(t: float) -> float:
        return cfg.noise_slow_level * np.interp(t, t_eval, slow) + cfg.noise_rf_level * np.interp(t, t_eval, rf)

    def control(t: float, kappa: float) -> np.ndarray:
        phase = kappa * (t - cfg.center)
        return j_peak * np.interp(t, t_eval, env) * (np.cos(phase) * SX_SUB + np.sin(phase) * SY_SUB)

    def noise(t: float) -> np.ndarray:
        return 0.5 * noise_delta(t) * SZ_SUB

    return t_eval, lambda k: lambda t: control(t, k) + noise(t), lambda k: lambda t: control(t_end - t, k) + noise(t_end - t), lambda k: lambda t: control(t_end - t, k) + noise(t)

def compute_A(kappa: float, cfg: WitnessConfig, seeds=None) -> dict[str, float]:
    if seeds is None: seeds = range(cfg.n_mc)
    a_exact, a_phys, f_forward, f_reversed = [], [], [], []
    t_eval = np.linspace(*cfg.t_span, cfg.n_points)
    
    for seed in seeds:
        _, h_f, h_er, h_pr = build_experiment(cfg, seed)
        
        psi_f = evolve(KET01, h_f(kappa), cfg.t_span, t_eval)[-1]
        psi_e = evolve(KET01, h_er(kappa), cfg.t_span, t_eval)[-1]
        psi_p = evolve(KET01, h_pr(kappa), cfg.t_span, t_eval)[-1]
        
        f0 = bell_fidelity(psi_f)
        f_e = bell_fidelity(psi_e)
        f_p = bell_fidelity(psi_p)
        
        f_forward.append(f0)
        f_reversed.append(f_p)
        a_exact.append(f0 - f_e)
        a_phys.append(f0 - f_p)
        
    a_exact = np.array(a_exact)
    a_phys = np.array(a_phys)
    return {
        "kappa": float(kappa),
        "A_exact_abs_max": float(np.max(np.abs(a_exact))),
        "A_phys_mean": float(np.mean(a_phys)),
        "A_phys_std": float(np.std(a_phys, ddof=1)),
        "A_phys_abs_max": float(np.max(np.abs(a_phys))),
        "F_forward_mean": float(np.mean(f_forward)),
        "F_reversed_mean": float(np.mean(f_reversed)),
    }

def run_witness(cfg: WitnessConfig = WitnessConfig()):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    print("="*50)
    print("TEST DU TÉMOIN D'INVERSION TEMPORELLE")
    print("="*50)
    
    for kappa in cfg.kappas:
        stat = compute_A(kappa, cfg)
        results[f"{kappa:.6g}"] = stat
        print(f"kappa={kappa:6.3f} | A_exact={stat['A_exact_abs_max']:.2e} | A_phys={stat['A_phys_mean']:.3e} ± {stat['A_phys_std']:.3e}")

    (cfg.out_dir / "rapport_temoin_inversion.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_witness(cfg, results)
    print(f"\nSUCCÈS : Figure sauvegardée dans {cfg.out_dir.resolve()}")

def _plot_witness(cfg: WitnessConfig, results: dict):
    kappas = np.array([float(k) for k in results])
    a_exact = np.array([results[k]["A_exact_abs_max"] for k in results])
    a_phys = np.array([abs(results[k]["A_phys_mean"]) for k in results])
    a_err = np.array([results[k]["A_phys_std"] for k in results])
    
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.errorbar(kappas, a_phys, yerr=a_err, marker="o", capsize=3, label=r"Bruit forward")
    ax.semilogy(kappas, np.maximum(a_exact, 1e-18), marker="s", linestyle="--", label=r"Contrôle exact")
    ax.set(title="Témoin d’inversion temporelle", xlabel=r"Chirp $\kappa$ (rad/ns$^2$)", ylabel="Asymétrie Bell")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "05_temoin_inversion_bruit.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    run_witness()
