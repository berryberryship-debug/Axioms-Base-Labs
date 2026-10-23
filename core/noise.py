#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QuantumLab — Générateur de Bruit Coloré (1/f) Reproducible."""

from typing import Optional
import numpy as np


def generate_1f_noise(
    n_samples: int, dt: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Génère un bruit de type 1/f (bruit rose) à moyenne nulle et variance unitaire.

    Paramètres
    ----------
    n_samples : int
        Nombre d'échantillons à générer (doit être >= 2).
    dt : float, optionnel
        Pas de temps d'échantillonnage (par défaut 1.0).
    seed : int, optionnel
        Graine pour le générateur de nombres pseudo-aléatoires.

    Retour
    ------
    np.ndarray
        Signal temporel 1D de taille (n_samples,).
    """
    if n_samples < 2:
        raise ValueError("n_samples doit être supérieur ou égal à 2.")

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)

    # Fréquences positives via RFFT
    f = np.fft.rfftfreq(n_samples, d=dt)

    # Filtre en amplitude : |H(f)| = 1 / sqrt(f)
    s_filter = np.zeros_like(f)
    s_filter[1:] = 1.0 / np.sqrt(f[1:])

    # Filtrage fréquentiel et transformation inverse
    noise_fft = np.fft.rfft(white) * s_filter
    noise = np.fft.irfft(noise_fft, n=n_samples)

    # Normalisation : moyenne nulle et écart-type unitaire
    noise = noise - np.mean(noise)
    std_val = np.std(noise)
    if std_val > 0.0:
        noise = noise / std_val

    return noise
