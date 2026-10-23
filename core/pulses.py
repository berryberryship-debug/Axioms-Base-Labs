#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantumLab — Module d'Impulsions Arbitraires (Gaussian, DRAG, Sinus, Composite)
"""

import numpy as np

def gaussian(t: np.ndarray, t0: float, sigma: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.exp(-((t - t0)**2) / (2 * sigma**2))

def drag(t: np.ndarray, t0: float, sigma: float, amplitude: float = 1.0, beta: float = 0.0) -> np.ndarray:
    g = gaussian(t, t0, sigma, amplitude)
    dg = -g * (t - t0) / (sigma**2)
    return g + 1j * beta * dg

def sine(t: np.ndarray, t0: float, duration: float, amplitude: float = 1.0, freq: float = 1.0) -> np.ndarray:
    mask = (t >= t0) & (t <= t0 + duration)
    res = np.zeros_like(t, dtype=float)
    res[mask] = amplitude * np.sin(2 * np.pi * freq * (t[mask] - t0))
    return res

def composite(t: np.ndarray, segments: list) -> np.ndarray:
    pulse_val = np.zeros_like(t, dtype=complex)
    for func, kwargs in segments:
        pulse_val += func(t, **kwargs)
    return pulse_val
