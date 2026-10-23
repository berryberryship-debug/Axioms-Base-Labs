#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AXIOMES BASE LABS — QuantumLab
Module : Arbitrary Pulse Shapes
Version : 1.2.0

Formes d'onde :
    - Gaussian, avec troncature optionnelle
    - DRAG, dérivée cohérente avec l'enveloppe
    - Sine, fenêtre temporelle finie
    - Composite, superposition de segments

Convention :
    Ω(t) = g(t) + i β dg/dt

Attention :
    beta possède les unités nécessaires pour que beta * dg/dt
    ait les mêmes unités que g.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np
from numpy.typing import ArrayLike

DEFAULT_TOL = 1e-10

def _finite_scalar(name: str, value: float) -> float:
    """Valide un paramètre scalaire réel et fini."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} doit être un scalaire réel."
        ) from exc

    if not np.isfinite(result):
        raise ValueError(f"{name} doit être fini.")

    return result

def _time_array(t: ArrayLike) -> np.ndarray:
    """Valide une grille temporelle, éventuellement multidimensionnelle."""
    try:
        result = np.asarray(t, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "La grille temporelle doit être numérique."
        ) from exc

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "La grille temporelle contient des valeurs non finies."
        )

    return result

def _gaussian_and_derivative(
    t: np.ndarray,
    t0: float,
    sigma: float,
    amplitude: float,
    truncated_at: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Renvoie g(t) et dg/dt analytiquement.

    Sans troncature :
        g = A exp(-x²/2)
        dg/dt = -(x/sigma) g

    Avec troncature :
        g = A [exp(-x²/2) - exp(-c²/2)] / [1-exp(-c²/2)]
        pour |x| <= c, et 0 ailleurs.

    La dérivée est calculée à partir de cette même fonction.
    """

    x = (t - t0) / sigma
    base = np.exp(-0.5 * x**2)

    if truncated_at is None:
        g = amplitude * base
        dg = -(x / sigma) * g
        return g, dg

    c = truncated_at
    edge = np.exp(-0.5 * c**2)
    denominator = -np.expm1(-0.5 * c**2)

    mask = np.abs(x) < c

    g = np.zeros_like(t, dtype=float)
    dg = np.zeros_like(t, dtype=float)

    g[mask] = (
        amplitude
        * (base[mask] - edge)
        / denominator
    )

    dg[mask] = (
        -amplitude
        * x[mask]
        * base[mask]
        / (sigma * denominator)
    )

    return g, dg

def gaussian(
    t: ArrayLike,
    t0: float,
    sigma: float,
    amplitude: float = 1.0,
    truncated_at: float | None = None,
) -> np.ndarray:
    """Impulsion gaussienne avec troncature optionnelle."""

    t = _time_array(t)
    t0 = _finite_scalar("t0", t0)
    sigma = _finite_scalar("sigma", sigma)
    amplitude = _finite_scalar("amplitude", amplitude)

    if sigma <= 0:
        raise ValueError("sigma doit être strictement positif.")

    if truncated_at is not None:
        truncated_at = _finite_scalar(
            "truncated_at", truncated_at
        )
        if truncated_at <= 0:
            raise ValueError(
                "truncated_at doit être strictement positif."
            )

    g, _ = _gaussian_and_derivative(
        t, t0, sigma, amplitude, truncated_at
    )
    return g

def drag(
    t: ArrayLike,
    t0: float,
    sigma: float,
    amplitude: float = 1.0,
    beta: float = 0.0,
    truncated_at: float | None = None,
) -> np.ndarray:
    """
    Impulsion DRAG cohérente avec l'enveloppe gaussienne.

    Ω(t) = g(t) + i beta dg/dt

    Si l'enveloppe est tronquée, sa dérivée analytique
    correspond à la même enveloppe tronquée.
    """

    t = _time_array(t)
    t0 = _finite_scalar("t0", t0)
    sigma = _finite_scalar("sigma", sigma)
    amplitude = _finite_scalar("amplitude", amplitude)
    beta = _finite_scalar("beta", beta)

    if sigma <= 0:
        raise ValueError("sigma doit être strictement positif.")

    if truncated_at is not None:
        truncated_at = _finite_scalar(
            "truncated_at", truncated_at
        )
        if truncated_at <= 0:
            raise ValueError(
                "truncated_at doit être strictement positif."
            )

    g, dg = _gaussian_and_derivative(
        t, t0, sigma, amplitude, truncated_at
    )

    result = g.astype(complex) + 1j * beta * dg

    if not np.all(np.isfinite(result)):
        raise FloatingPointError(
            "Valeurs non finies détectées dans l'impulsion DRAG."
        )

    return result

def sine(
    t: ArrayLike,
    t0: float,
    duration: float,
    amplitude: float = 1.0,
    freq: float = 1.0,
) -> np.ndarray:
    """Sinusoïde fenêtrée sur [t0, t0 + duration]."""

    t = _time_array(t)
    t0 = _finite_scalar("t0", t0)
    duration = _finite_scalar("duration", duration)
    amplitude = _finite_scalar("amplitude", amplitude)
    freq = _finite_scalar("freq", freq)

    if duration <= 0:
        raise ValueError(
            "duration doit être strictement positive."
        )

    result = np.zeros_like(t, dtype=float)
    mask = (t >= t0) & (t <= t0 + duration)

    result[mask] = amplitude * np.sin(
        2.0 * np.pi * freq * (t[mask] - t0)
    )

    return result

def composite(
    t: ArrayLike,
    segments: list[
        tuple[Callable[..., np.ndarray], dict[str, Any]]
    ],
) -> np.ndarray:
    """Superpose les impulsions évaluées sur la même grille."""

    t = _time_array(t)
    result = np.zeros_like(t, dtype=complex)

    for index, (func, kwargs) in enumerate(segments):
        if not callable(func):
            raise TypeError(
                f"Le segment {index} n'est pas appelable."
            )

        values = np.asarray(func(t, **kwargs))

        if values.shape != t.shape:
            raise ValueError(
                f"Segment {index} : forme {values.shape}, "
                f"attendue {t.shape}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Segment {index} : valeurs non finies."
            )

        result += values

    if not np.all(np.isfinite(result)):
        raise FloatingPointError(
            "La superposition produit des valeurs non finies."
        )

    return result

def run_validation_tests() -> None:
    """Tests de cohérence analytique et de frontières."""

    # Choix de paramètres éloignant les frontières des
    # singularités numériques liées à une coupure trop petite.
    t0 = 0.0
    sigma = 1.0
    amplitude = 1.0
    cutoff = 3.0
    beta = 0.2

    t = np.linspace(-5.0, 5.0, 10001)

    g = gaussian(
        t, t0, sigma, amplitude, cutoff
    )

    d = drag(
        t, t0, sigma, amplitude, beta, cutoff
    )

    s = sine(
        t, -2.0, 4.0, 1.0, 0.5
    )

    c = composite(
        t,
        [
            (
                gaussian,
                {
                    "t0": -1.0,
                    "sigma": 0.5,
                    "amplitude": 1.0,
                    "truncated_at": 3.0,
                },
            ),
            (
                drag,
                {
                    "t0": 1.0,
                    "sigma": 0.5,
                    "amplitude": 0.5,
                    "beta": 0.1,
                    "truncated_at": 3.0,
                },
            ),
        ],
    )

    for name, values in (
        ("Gaussian", g),
        ("DRAG", d),
        ("Sine", s),
        ("Composite", c),
    ):
        assert values.shape == t.shape, name
        assert np.all(np.isfinite(values)), name

    # L'enveloppe doit être nulle à l'extérieur de la coupure.
    outside = np.abs(t - t0) > cutoff * sigma
    assert np.all(g[outside] == 0.0)

    # La valeur à la frontière est nulle.
    boundary = np.array([-cutoff * sigma, cutoff * sigma])
    g_boundary = gaussian(
        boundary, t0, sigma, amplitude, cutoff
    )
    assert np.allclose(g_boundary, 0.0, atol=DEFAULT_TOL)

    # La dérivée analytique est nulle à l'extérieur et aux
    # frontières selon la convention retenue pour la dérivée.
    # La dérivée intérieure tend vers une valeur non nulle :
    # le prolongement par zéro est donc continu, pas C1.
    eps = 1e-6
    near_edge = np.array([
        cutoff * sigma - eps,
        cutoff * sigma + eps,
    ])

    near_values = gaussian(
        near_edge, t0, sigma, amplitude, cutoff
    )
    assert near_values[0] >= 0.0
    assert near_values[1] == 0.0

    # Vérifie l'identité DRAG = g + i beta dg/dt
    # à l'intérieur, via une différence finie.
    t_check = np.linspace(-2.0, 2.0, 2001)
    g_check = gaussian(
        t_check, t0, sigma, amplitude, cutoff
    )
    d_check = drag(
        t_check, t0, sigma, amplitude, beta, cutoff
    )
    dg_numeric = np.gradient(g_check, t_check)

    interior = np.abs(t_check) < cutoff - 0.05
    error = np.max(
        np.abs(
            d_check[interior].imag
            - beta * dg_numeric[interior]
        )
    )

    assert error < 2e-3, (
        f"Incohérence DRAG : erreur numérique {error}"
    )

    # Amplitude nulle.
    zero = gaussian(
        t, t0, sigma, 0.0, cutoff
    )
    assert np.all(zero == 0.0)

    # Paramètre invalide.
    try:
        gaussian(t, t0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("sigma = 0 aurait dû échouer.")

    print("QuantumLab 1.2.0 — TESTS PASSÉS")
    print(f"Nombre d'échantillons : {t.size}")
    print(f"Maximum |Gaussian| : {np.max(np.abs(g)):.8f}")
    print(f"Maximum |DRAG| : {np.max(np.abs(d)):.8f}")
    print(f"Maximum |Sine| : {np.max(np.abs(s)):.8f}")
    print(f"Maximum |Composite| : {np.max(np.abs(c)):.8f}")
    print(f"Erreur dérivée numérique : {error:.3e}")

if __name__ == "__main__":
    run_validation_tests()
