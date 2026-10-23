#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuantumLab — Phase 3
Champ Moiré + échantillonnage spatial.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MoireField:
    k1: tuple[float, float]
    k2: tuple[float, float]
    amplitude: float = 1.0
    phi1: float = 0.0
    phi2: float = 0.0
    temporal_frequency: float = 0.0
    temporal_phase: float = 0.0

    def sample(self, x, y, t=0.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        t = np.asarray(t, dtype=float)

        k1x, k1y = self.k1
        k2x, k2y = self.k2

        phase1 = k1x * x + k1y * y + self.phi1
        phase2 = k2x * x + k2y * y + self.phi2

        spatial_field = np.cos(phase1) + np.cos(phase2)

        if self.temporal_frequency != 0.0:
            temporal_factor = np.cos(2.0 * np.pi * self.temporal_frequency * t + self.temporal_phase)
        else:
            temporal_factor = 1.0

        return self.amplitude * spatial_field * temporal_factor

def gaussian_weight(x, y, x0, y0, sigma_device):
    if sigma_device <= 0.0:
        raise ValueError("sigma_device doit être > 0.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return np.exp(-r2 / (2.0 * sigma_device ** 2))

def effective_field(field, x, y, t, *, x0, y0, sigma_device):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x et y doivent être des grilles 2D.")
    if x.shape != y.shape:
        raise ValueError("x et y doivent avoir la même forme.")

    weights = gaussian_weight(x, y, x0, y0, sigma_device)
    values = field.sample(x, y, t)
    denominator = np.sum(weights)
    if denominator == 0.0:
        raise ValueError("Normalisation spatiale nulle.")
    return float(np.sum(weights * values) / denominator)
