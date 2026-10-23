#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def state_fidelity(psi_target: np.ndarray, psi_achieved: np.ndarray) -> float:
    overlap = np.vdot(psi_target, psi_achieved)
    return float(np.abs(overlap)**2)

def infidelity(psi_target: np.ndarray, psi_achieved: np.ndarray) -> float:
    return 1.0 - state_fidelity(psi_target, psi_achieved)
