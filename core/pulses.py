#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QuantumLab — Module d'Impulsions Arbitraires (Gaussian, DRAG, Sinus, Composite)."""

from __future__ import annotations
from typing import Callable, Any
import numpy as np


def gaussian(
    t: np.ndarray,
    t0: float,
    sigma: float,
    amplitude: float = 1.0,
    truncated_at: float | None = None,
) -> np.ndarray:
    """Impulsion gaussienne avec option de troncature lisse."""
    if sigma <= 0.0:
        raise ValueError("sigma doit être strictement positif.")

    t = np.asarray(t, dtype=float)
    g = amplitude * np.exp(-((t - t0) ** 2) / (2.0 * sigma**2))

    if truncated_at is not None:
        cutoff = truncated_at * sigma
        mask = np.abs(t - t0) <= cutoff
        offset = amplitude * np.exp(-(cutoff**2) / (2.0 * sigma**2))
        g = np.where(mask, (g - offset) / (1.0 - offset / amplitude), 0.0)

    return g


def drag(
    t: np.ndarray,
    t0: float,
    sigma: float,
    amplitude: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Impulsion DRAG : composante en phase (gaussienne) et en quadrature (dérivée)."""
    if sigma <= 0.0:
        raise ValueError("sigma doit être strictement positif.")

    t = np.asarray(t, dtype=float)
    g = gaussian(t, t0, sigma, amplitude)
    dg = -g * (t - t0) / (sigma**2)

    return g + 1j * beta * dg


def sine(
    t: np.ndarray,
    t0: float,
    duration: float,
    amplitude: float = 1.0,
    freq: float = 1.0,
) -> np.ndarray:
    """Créneau sinusoïdal sur l'intervalle [t0, t0 + duration]."""
    if duration <= 0.0:
        raise ValueError("duration doit être strictement positive.")

    t = np.asarray(t, dtype=float)
    res = np.zeros_like(t, dtype=float)
    mask = (t >= t0) & (t <= t0 + duration)
    res[mask] = amplitude * np.sin(2.0 * np.pi * freq * (t[mask] - t0))
    return res


def composite(
    t: np.ndarray, segments: list[tuple[Callable[..., np.ndarray], dict[str, Any]]]
) -> np.ndarray:
    """Superpose plusieurs formes d'onde sur la même grille temporelle."""
    t = np.asarray(t, dtype=float)
    pulse_val = np.zeros_like(t, dtype=complex)
    for func, kwargs in segments:
        pulse_val += func(t, **kwargs)
    return pulse_val
