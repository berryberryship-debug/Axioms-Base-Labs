#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantumLab — Phase 3
Champ Moiré + échantillonnage spatial.
Version révisée : validation des entrées et stabilité numérique.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _finite_scalar(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} doit être un scalaire numérique fini.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} doit être fini.")
    return result


def _vector2(name, value):
    try:
        if len(value) != 2:
            raise ValueError
        return (
            _finite_scalar(f"{name}[0]", value[0]),
            _finite_scalar(f"{name}[1]", value[1]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} doit contenir deux composantes finies.") from exc


def _finite_array(name, value):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} doit être numérique.") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} doit contenir uniquement des valeurs finies.")
    return result


@dataclass(frozen=True)
class MoireField:
    """Somme de deux ondes planes modulée dans le temps."""

    k1: tuple[float, float]
    k2: tuple[float, float]
    amplitude: float = 1.0
    phi1: float = 0.0
    phi2: float = 0.0
    temporal_frequency: float = 0.0
    temporal_phase: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "k1", _vector2("k1", self.k1))
        object.__setattr__(self, "k2", _vector2("k2", self.k2))

        for name in (
            "amplitude", "phi1", "phi2",
            "temporal_frequency", "temporal_phase"
        ):
            object.__setattr__(
                self, name, _finite_scalar(name, getattr(self, name))
            )

        if self.temporal_frequency < 0:
            raise ValueError("temporal_frequency doit être >= 0.")

    def sample(self, x, y, t=0.0):
        """Échantillonne le champ avec broadcasting NumPy."""
        x, y, t = np.broadcast_arrays(
            _finite_array("x", x),
            _finite_array("y", y),
            _finite_array("t", t),
        )

        k1x, k1y = self.k1
        k2x, k2y = self.k2

        phase1 = k1x * x + k1y * y + self.phi1
        phase2 = k2x * x + k2y * y + self.phi2

        spatial_field = np.cos(phase1) + np.cos(phase2)

        if self.temporal_frequency == 0.0:
            temporal_factor = 1.0
        else:
            temporal_factor = np.cos(
                2.0 * np.pi * self.temporal_frequency * t
                + self.temporal_phase
            )

        result = self.amplitude * spatial_field * temporal_factor

        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Le champ calculé contient une valeur non finie.")

        return np.asarray(result, dtype=float)


def gaussian_weight(x, y, x0, y0, sigma_device):
    """Calcule les poids gaussiens spatiaux non normalisés."""
    x0 = _finite_scalar("x0", x0)
    y0 = _finite_scalar("y0", y0)
    sigma_device = _finite_scalar("sigma_device", sigma_device)

    if sigma_device <= 0:
        raise ValueError("sigma_device doit être > 0.")

    x, y = np.broadcast_arrays(
        _finite_array("x", x),
        _finite_array("y", y),
    )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        dx = (x - x0) / sigma_device
        dy = (y - y0) / sigma_device
        r2 = dx * dx + dy * dy
        weights = np.exp(-0.5 * r2)

    if np.any(np.isnan(weights)):
        raise FloatingPointError("Le calcul gaussien a produit NaN.")

    return weights


def effective_field(
    field, x, y, t=0.0, *, x0, y0, sigma_device
):
    """
    Moyenne discrète du champ pondérée par une gaussienne.

    x et y doivent être des grilles 2D de même forme.
    t peut être un scalaire, une grille 2D compatible ou un tableau
    de temps 1D contenant plusieurs instants.
    """
    x = _finite_array("x", x)
    y = _finite_array("y", y)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x et y doivent être des grilles 2D.")

    if x.shape != y.shape:
        raise ValueError("x et y doivent avoir la même forme.")

    if x.size == 0:
        raise ValueError("Les grilles ne doivent pas être vides.")

    weights = gaussian_weight(x, y, x0, y0, sigma_device)
    weight_max = float(np.max(weights))

    if not math.isfinite(weight_max) or weight_max <= 0:
        raise ValueError(
            "Tous les poids sont nuls numériquement. "
            "Élargir sigma_device ou rapprocher le centre de la grille."
        )

    weights = weights / weight_max
    denominator = float(np.sum(weights, dtype=np.float64))

    if not math.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("Normalisation spatiale invalide.")

    t_array = _finite_array("t", t)

    if t_array.ndim == 1 and t_array.size > 1:
        values = field.sample(
            x[None, :, :],
            y[None, :, :],
            t_array[:, None, None],
        )
        numerator = np.sum(
            values * weights[None, :, :],
            axis=(-2, -1),
            dtype=np.float64,
        )
        result = np.asarray(numerator / denominator, dtype=float)

    else:
        values = field.sample(x, y, t_array)

        try:
            weighted = weights * values
        except ValueError as exc:
            raise ValueError(
                "La forme de t n'est pas compatible avec la grille spatiale."
            ) from exc

        if weighted.ndim >= 2 and weighted.shape[-2:] == x.shape:
            numerator = np.sum(
                weighted, axis=(-2, -1), dtype=np.float64
            )
        else:
            raise ValueError(
                "t doit être scalaire, une grille 2D compatible "
                "ou un tableau de temps 1D."
            )

        result = np.asarray(numerator / denominator, dtype=float)

    if not np.all(np.isfinite(result)):
        raise FloatingPointError("La moyenne effective n'est pas finie.")

    return float(result) if result.ndim == 0 else result
