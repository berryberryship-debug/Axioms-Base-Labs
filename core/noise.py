#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantumLab — Générateur de Bruit Coloré (1/f) Reproducible
"""

import numpy as np

def generate_1f_noise(n_samples: int, dt: float = 1.0, seed: int = None) -> np.ndarray:
    """
    Génère un bruit de type 1/f (bruit rose) stationnaire et reproductible par seed.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génération d'un bruit blanc gaussien initial
    white = np.random.randn(n_samples)
    
    # Fréquences associées via FFT
    f = np.fft.rfftfreq(n_samples, d=dt)
    f[0] = f[1]  # Éviter la singularité en f=0
    
    # Filtre spectral pour obtenir une densité spectrale de puissance en 1/f
    # Amplitude spectrale proportionnelle à 1/sqrt(f)
    s_filter = 1.0 / np.sqrt(f)
    s_filter[0] = 0.0
    
    noise_fft = np.fft.rfft(white) * s_filter
    noise = np.fft.irfft(noise_fft, n=n_samples)
    
    # Normalisation de l'écart-type
    noise = noise / (np.std(noise) + 1e-12)
    return noise
