"""
V100 — Dispersion des Magnons 1D (Ondes de spin)
Exploration paramétrique du couplage J avec calcul rigoureux de m*.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMÈTRES PHYSIQUES ET CONSTANTES
# ==========================================
S = 0.5
a = 1.0  # Å
J_LIST = [0.5, 1.0, 2.0]  # meV

# Constantes physiques fondamentales (CODATA / SI)
HBAR_SI = 1.054571817e-34       # J·s
HBAR_MEV_S = 6.582119569e-13    # meV·s
M_E = 9.1093837e-31             # kg
MEV_TO_J = 1.602176634e-22      # 1 meV en Joules
A_TO_M = 1e-10                  # 1 Å en mètres

# ==========================================
# 2. FONCTION DE DISPERSION ET MASSE EFFECTIVE
# ==========================================
def calculer_dispersion(J, S, a, N=400):
    q = np.linspace(-np.pi/a, np.pi/a, N)
    # Formule stable numériquement en q=0
    E_q = 4 * J * S * np.sin(q * a / 2.0)**2
    # Pente énergétique et vitesse de groupe
    dE_dq = 2 * J * S * a * np.sin(q * a)
    v_g = (dE_dq / HBAR_MEV_S) * A_TO_M  # m/s
    
    # Masse effective au point Gamma : m* = hbar^2 / (2 * J * S * a^2)
    d2E_dq2_SI = 2 * (J * MEV_TO_J) * S * (a * A_TO_M)**2
    m_eff_kg = (HBAR_SI**2) / d2E_dq2_SI
    m_eff_ratio_me = m_eff_kg / M_E
    
    return q, E_q, v_g, m_eff_kg, m_eff_ratio_me

# ==========================================
# 3. EXÉCUTION ET AFFICHAGE
# ==========================================
print("======================================================================")
print("V100 — ÉTUDE COMPARATIVE DU COUPLAGE D'ÉCHANGE J (FERROMAGNÉTIQUE 1D)")
print("======================================================================")

donnees = {}
for J in J_LIST:
    q, E_q, v_g, m_eff_kg, m_eff_ratio = calculer_dispersion(J, S, a)
    donnees[J] = (q, E_q, v_g)
    print(f"J = {J:3.1f} meV | E_max = {E_q.max():5.2f} meV | v_g,max = {np.max(np.abs(v_g)):6.2f} m/s | m* = {m_eff_kg:.3e} kg ({m_eff_ratio:7.1f} m_e)")
print("======================================================================")

# ==========================================
# 4. VISUALISATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for J, (q, E_q, v_g) in donnees.items():
    ax1.plot(q, E_q, lw=2, label=f"J = {J} meV")
    ax2.plot(q, v_g, lw=2, label=f"J = {J} meV")

# Panneau 1 : Dispersion
ax1.set_title("Dispersion \(E(q)\) pour différents couplages")
ax1.set_xlabel("Vecteur d'onde \(q\) (Å\(^{-1}\))")
ax1.set_ylabel(r"Énergie \(E(q)\) (meV)")
ax1.set_xticks([-np.pi/a, 0, np.pi/a])
ax1.set_xticklabels([r'\(-\pi/a\)', r'\(0\) (\(\Gamma\))', r'\(\pi/a\)'])
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panneau 2 : Vitesse de groupe
ax2.set_title(r"Vitesse de groupe \(v_g(q)\)")
ax2.set_xlabel("Vecteur d'onde \(q\) (Å\(^{-1}\))")
ax2.set_ylabel(r"Vitesse \(v_g(q)\) (m/s)")
ax2.set_xticks([-np.pi/a, 0, np.pi/a])
ax2.set_xticklabels([r'\(-\pi/a\)', r'\(0\) (\(\Gamma\))', r'\(\pi/a\)'])
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("v100_magnon_dispersion.png", dpi=150)
print("Figure sauvegardée : v100_magnon_dispersion.png")
