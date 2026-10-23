#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AXIOMES BASE LABS — Master Pipeline
Version : 1.2.0

Orchestration :
    - Calibration d'une impulsion gaussienne
    - Évolution quantique sans bruit
    - Évolution quantique avec bruit 1/f (Monte-Carlo)
    - Mesures de fidélité et d'infidélité
    - Contrôles numériques et rapport final

Convention :
    hbar = 1
    H(t) = hbar * Omega(t) / 2 * sigma_x
"""

from __future__ import annotations

import sys
import numpy as np

from core.quantum_core import evolve, qubit_hamiltonian
from core.pulses import gaussian
from core.noise import generate_1f_noise
from core.metrics import state_fidelity, infidelity

def run_simulation() -> int:
    print("=" * 58)
    print("AXIOMES BASE LABS — MASTER PIPELINE 1.2.0")
    print("=" * 58)

    # --------------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------------
    hbar = 1.0
    t_span = (0.0, 20.0)
    t_eval = np.linspace(*t_span, 2001)
    dt = float(t_eval[1] - t_eval[0])

    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    target_state = np.array([0.0 + 0.0j, 1.0 + 0.0j])

    center = 0.5 * (t_span[0] + t_span[1])
    sigma = 2.0

    # --------------------------------------------------------
    # 2. Enveloppe et calibration de l'aire (Point 1: Compatibilité)
    # --------------------------------------------------------
    envelope = np.asarray(
        gaussian(t_eval, t0=center, sigma=sigma, amplitude=1.0),
        dtype=float,
    )

    if envelope.shape != t_eval.shape:
        raise ValueError("gaussian() doit retourner un tableau de même taille.")
    if not np.all(np.isfinite(envelope)):
        raise ValueError("L'enveloppe contient des valeurs non finies.")

    # Compatibilité NumPy 2.0 (trapezoid) et 1.x (trapz)
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_func is None:
        raise RuntimeError("Aucune fonction trapèze trouvée dans NumPy.")

    envelope_area = float(trapz_func(envelope, t_eval))
    if not np.isfinite(envelope_area) or envelope_area <= 0.0:
        raise ValueError("Aire d'impulsion invalide.")

    target_rotation = np.pi
    omega_peak = target_rotation / envelope_area
    calibrated_area = float(trapz_func(omega_peak * envelope, t_eval))

    print(f"[INFO] Aire initiale       : {envelope_area:.8f}")
    print(f"[INFO] Amplitude calibrée  : {omega_peak:.8f}")
    print(f"[INFO] Rotation calibrée   : {calibrated_area:.8f} rad")

    # --------------------------------------------------------
    # 3. Hamiltonien sans bruit (Point 2: Scalaire direct)
    # --------------------------------------------------------
    def hamiltonian_clean(t: float) -> np.ndarray:
        # Calcul scalaire direct (évite la création de np.array([t]))
        amplitude = np.exp(-((t - center) ** 2) / (2.0 * sigma ** 2))
        
        return qubit_hamiltonian(
            delta=0.0,
            omega_x=omega_peak * amplitude,
            omega_y=0.0,
            hbar=hbar,
        )

    # --------------------------------------------------------
    # 4. Simulation de référence
    # --------------------------------------------------------
    print("\n[+] Simulation sans bruit...")
    clean = evolve(
        psi0=psi0,
        hamiltonian=hamiltonian_clean,
        t_span=t_span,
        t_eval=t_eval,
        hbar=hbar,
        rtol=1e-10,
        atol=1e-12,
    )

    fidelity_clean = state_fidelity(target_state, clean.final_state)
    infidelity_clean = infidelity(target_state, clean.final_state)

    # --------------------------------------------------------
    # 5. Simulation avec bruit 1/f (Point 3: Monte-Carlo)
    # --------------------------------------------------------
    print("[+] Simulation avec bruit 1/f (Monte-Carlo N=100)...")
    
    n_runs = 100
    noise_amplitude = 0.05
    fidelities_noisy = []
    norm_errors = []

    for seed in range(n_runs):
        noise_raw = np.asarray(
            generate_1f_noise(n_samples=len(t_eval), dt=dt, seed=seed),
            dtype=float,
        )
        
        # Normalisation stricte du bruit (moyenne 0, écart-type 1)
        noise = (noise_raw - np.mean(noise_raw)) / np.std(noise_raw)

        def hamiltonian_noisy(t: float) -> np.ndarray:
            # Interpolation du bruit à l'instant t
            noise_t = float(np.interp(t, t_eval, noise))
            
            # Calcul scalaire direct
            envelope_t = np.exp(-((t - center) ** 2) / (2.0 * sigma ** 2))
            omega_t = omega_peak * envelope_t * (1.0 + noise_amplitude * noise_t)
            
            return qubit_hamiltonian(
                delta=0.0,
                omega_x=omega_t,
                omega_y=0.0,
                hbar=hbar,
            )

        noisy = evolve(
            psi0=psi0,
            hamiltonian=hamiltonian_noisy,
            t_span=t_span,
            t_eval=t_eval,
            hbar=hbar,
            rtol=1e-10,
            atol=1e-12,
        )

        fid = state_fidelity(target_state, noisy.final_state)
        fidelities_noisy.append(fid)
        norm_errors.append(noisy.max_norm_error)

    # Statistiques de l'ensemble
    fid_mean = float(np.mean(fidelities_noisy))
    fid_std = float(np.std(fidelities_noisy))
    max_norm_err_noisy = float(np.max(norm_errors))
    infid_mean = 1.0 - fid_mean

    # --------------------------------------------------------
    # 6. Rapport
    # --------------------------------------------------------
    print("\n" + "=" * 58)
    print("RAPPORT DE SIMULATION (MONTE-CARLO)")
    print("=" * 58)

    print(f"Norme max. erreur — propre : {clean.max_norm_error:.3e}")
    print(f"Norme max. erreur — bruit  : {max_norm_err_noisy:.3e}")

    print(f"Fidélité sans bruit        : {fidelity_clean:.10f}")
    print(f"Infidélité sans bruit      : {infidelity_clean:.3e}")

    print(f"Fidélité moyenne (N=100)   : {fid_mean:.10f} ± {fid_std:.3e}")
    print(f"Infidélité moyenne         : {infid_mean:.3e}")
    print(f"Variation de fidélité      : {fid_mean - fidelity_clean:+.3e}")
    print(f"Écart-type relatif du bruit: {noise_amplitude:.3f}")
    print("=" * 58)

    # --------------------------------------------------------
    # 7. Contrôles d'intégrité
    # --------------------------------------------------------
    checks = {
        "Fidélité propre finie": np.isfinite(fidelity_clean),
        "Fidélité moyenne finie": np.isfinite(fid_mean),
        "Fidélité propre dans [0,1]": (-1e-10 <= fidelity_clean <= 1.0 + 1e-10),
        "Fidélité moyenne dans [0,1]": (-1e-10 <= fid_mean <= 1.0 + 1e-10),
        "Norme conservée sans bruit": (clean.max_norm_error < 1e-7),
        "Norme conservée avec bruit": (max_norm_err_noisy < 1e-7),
    }

    print("\nCONTRÔLES")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if all(checks.values()):
        print("\nPASS NUMÉRIQUE : les contrôles minimaux sont satisfaits.")
        print("Cela ne constitue pas une validation expérimentale.")
        return 0

    print("\nFAIL : au moins un contrôle numérique a échoué.")
    return 1

if __name__ == "__main__":
    sys.exit(run_simulation())
