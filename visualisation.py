#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXIOMES BASE LABS — Module de Visualisation Scientifique
Version : 1.1.0
"""
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from core.quantum_core import evolve, qubit_hamiltonian
from core.pulses import gaussian
from core.noise import generate_1f_noise
from core.metrics import state_fidelity

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

@dataclass
class SimConfig:
    t_span: tuple = (0.0, 20.0)
    n_points: int = 1000
    center: float = 10.0
    sigma: float = 2.0
    n_mc: int = 100
    noise_level: float = 0.05
    out_dir: Path = Path("figures")

def generate_all_figures(cfg: SimConfig = SimConfig()):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("="*50)
    log.info("GÉNÉRATION DES FIGURES SCIENTIFIQUES - AXIOMES")
    log.info("="*50)

    t_eval = np.linspace(*cfg.t_span, cfg.n_points)
    dt = float(t_eval[1] - t_eval[0])
    psi0 = np.array([1.0+0.0j, 0.0+0.0j])
    target = np.array([0.0+0.0j, 1.0+0.0j])

    # Enveloppe de référence - calculée UNE SEULE FOIS
    envelope = gaussian(t_eval, t0=cfg.center, sigma=cfg.sigma, amplitude=1.0)
    area = np.trapezoid(envelope, t_eval)
    omega_peak = np.pi / area  # Condition porte X

    # Helper pour interpoler l'enveloppe dans le hamiltonien
    def make_hamiltonian(noise_arr=None):
        def h_func(t: float):
            env_t = np.interp(t, t_eval, envelope)
            if noise_arr is not None:
                nt = np.interp(t, t_eval, noise_arr)
                env_t *= (1.0 + cfg.noise_level * nt)
            return qubit_hamiltonian(delta=0.0, omega_x=omega_peak * env_t)
        return h_func

    # --- FIGURE 1 ---
    log.info("[1/3] Impulsion de contrôle...")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_eval, envelope, linewidth=2.5)
    ax.set(title="Impulsion de contrôle (Gaussienne)", xlabel="Temps (ns)", ylabel="Amplitude (a.u.)")
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "01_impulsion.png", dpi=300)
    plt.close(fig)

    # --- FIGURE 2 ---
    log.info("[2/3] Évolution temporelle...")
    states = evolve(psi0, make_hamiltonian(), cfg.t_span, t_eval)
    pop_0, pop_1 = np.abs(states[:,0])**2, np.abs(states[:,1])**2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_eval, pop_0, label="|0>", linewidth=2)
    ax.plot(t_eval, pop_1, label="|1>", linewidth=2)
    ax.set(title="Évolution du Qubit (Rotation X - π pulse)", xlabel="Temps (ns)", ylabel="Population")
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "02_evolution.png", dpi=300)
    plt.close(fig)

    # --- FIGURE 3 ---
    log.info(f"[3/3] Monte-Carlo robustesse (N={cfg.n_mc})...")
    fids = []
    for seed in range(cfg.n_mc):
        noise = generate_1f_noise(len(t_eval), dt, seed)
        noise = (noise - np.mean(noise)) / (np.std(noise) + 1e-12)
        states_n = evolve(psi0, make_hamiltonian(noise), cfg.t_span, t_eval)
        fids.append(state_fidelity(target, states_n[-1]))

    fids = np.array(fids)
    log.info(f"  -> Fidélité moyenne: {np.mean(fids):.4f} ± {np.std(fids):.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fids, bins=20, edgecolor='black', alpha=0.8)
    ax.set(title=f"Distribution Fidélité sous Bruit 1/f (N={cfg.n_mc})", xlabel="Fidélité", ylabel="Occurrences")
    ax.axvline(np.mean(fids), linestyle='--', label=f"Moyenne {np.mean(fids):.3f}")
    ax.legend(); ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "03_monte_carlo.png", dpi=300)
    plt.close(fig)

    log.info("SUCCÈS : Figures dans %s", cfg.out_dir.resolve())

if __name__ == "__main__":
    generate_all_figures()
