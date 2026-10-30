"""
V102 — Dispersion des Magnons avec DMI : Stabilité, Champ critique et Spirale
Comparaison des régimes FM instable, FM stabilisé et fondamental spiralé.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMÈTRES PHYSIQUES UNIFIÉS
# ==========================================
J = 1.0       # Échange isotrope (meV)
D = 0.5       # Constante DMI (meV)
S = 0.5       # Spin
a = 1.0       # Paramètre de maille (Å)
HBAR_MEV_S = 6.582119569e-13
A_TO_M = 1e-10

# Champ critique de saturation (pour stabiliser l'état colinéaire)
h_c = 2 * S * (np.sqrt(J**2 + D**2) - J)
h_stab = 1.2 * h_c  # Champ appliqué pour forcer la stabilité

# ==========================================
# 2. VECTEUR D'ONDE ET DISPERSIONS
# ==========================================
N = 1000
q = np.linspace(-np.pi / a, np.pi / a, N)

# Régime 1 : Ferromagnétique colinéaire (sans champ, instable)
# E(q) < 0 autour du minimum -> mode mou
E_fm_sans_h = 2 * J * S * (1 - np.cos(q * a)) + 2 * D * S * np.sin(q * a)

# Régime 2 : Ferromagnétique stabilisé par champ externe (h > h_c)
E_fm_stab = E_fm_sans_h + h_stab

# Régime 3 : Fondamental spiralé (h = 0)
# La DMI crée une spirale avec Q = arctan(D/J)/a.
# Le mode de Goldstone (énergie nulle) est à q=0.
J_eff = np.sqrt(J**2 + D**2)
E_spiral = 2 * S * J_eff * np.abs(np.sin(q * a / 2.0))

# Grandeurs de non-réciprocité (chiralité) pour l'état ferromagnétique
Delta_E = 4 * D * S * np.sin(q * a)
denom = 4 * J * S * (1 - np.cos(q * a)) + 2 * h_stab
C_q = np.divide(Delta_E, denom, out=np.zeros_like(Delta_E), where=denom > 1e-6)

# ==========================================
# 3. EXTRACTION DES MINIMA ET AFFICHAGE
# ==========================================
q_min_theo = -np.arctan(D / J) / a
E_min_theo_stab = 2 * J * S - 2 * S * J_eff + h_stab

idx_min = np.argmin(E_fm_stab)
q_min_num = q[idx_min]
E_min_num = np.min(E_fm_stab)

print("==================================================")
print("V102 — ANALYSE DE STABILITÉ ET CHIRALITÉ (DMI)")
print("==================================================")
print(f"Champ critique de saturation h_c : {h_c:.4f} meV")
print(f"Champ stabilisateur appliqué   : {h_stab:.4f} meV")
print(f"q_min théorique (FM stabilisé) : {q_min_theo:.6f} Å⁻¹")
print(f"q_min numérique (FM stabilisé) : {q_min_num:.6f} Å⁻¹")
print(f"E_min théorique (FM stabilisé) : {E_min_theo_stab:.6f} meV")
print(f"E_min numérique (FM stabilisé) : {E_min_num:.6f} meV")
print(f"Mode de Goldstone (Spirale)    : E(0) = {E_spiral[N//2]:.6f} meV")
print("==================================================")

# ==========================================
# 4. VISUALISATION
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panneau 1 : Comparaison des régimes
axes[0].plot(q, E_fm_sans_h, 'r--', lw=1.8, label="FM instable (h=0)")
axes[0].plot(q, E_fm_stab, 'b-', lw=2, label=f"FM stabilisé (h={h_stab:.2f} meV)")
axes[0].plot(q, E_spiral, 'g-', lw=2, label="Spirale (mode Goldstone)")
axes[0].axhline(0, color='k', linestyle=':', alpha=0.6)
axes[0].axvline(q_min_theo, color='blue', linestyle=':', label=f"q_min = {q_min_theo:.2f} Å⁻¹")
axes[0].set_title("Spectres d'excitation E(q)")
axes[0].set_xlabel(r"Vecteur d'onde q (Å\(^{-1}\))")
axes[0].set_ylabel(r"Énergie (meV)")
axes[0].legend(loc="upper right", fontsize=9)

# Panneau 2 : Asymétrie absolue
axes[1].plot(q, Delta_E, 'purple', lw=2, label=r"\(\Delta E(q) = 4DS\sin(qa)\)")
axes[1].axhline(0, color='k', linestyle=':', alpha=0.5)
axes[1].set_title("Non-réciprocité absolue \(\Delta E(q)\)")
axes[1].set_xlabel(r"Vecteur d'onde q (Å\(^{-1}\))")
axes[1].set_ylabel(r"\(\Delta E\) (meV)")
axes[1].legend(loc="upper right", fontsize=9)

# Panneau 3 : Asymétrie relative
axes[2].plot(q, C_q, 'darkgreen', lw=2, label=r"\(\mathcal{C}(q) = \Delta E / E_{total}\)")
axes[2].axhline(0, color='k', linestyle=':', alpha=0.5)
axes[2].set_title("Asymétrie relative \(\mathcal{C}(q)\)")
axes[2].set_xlabel(r"Vecteur d'onde q (Å\(^{-1}\))")
axes[2].set_ylabel(r"Chiralité \(\mathcal{C}\)")
axes[2].legend(loc="upper right", fontsize=9)

for ax in axes:
    ax.set_xticks([-np.pi / a, 0, np.pi / a])
    ax.set_xticklabels([r'\(-\pi/a\)', r'\(0\)', r'\(\pi/a\)'])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("v102_magnon_dmi.png", dpi=150)
print("Figure sauvegardée : v102_magnon_dmi.png")
