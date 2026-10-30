"""
V101 — Propagation Dynamique d'un Paquet d'Ondes de Spin (Magnons 1D)
Chaîne discrète périodique avec intégration spectrale exacte (FFT).
Audit renforcé : énergie calculée sur l'état propagé (pas sur l'analytique),
et surveillance dynamique de la périodicité du centroïde.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTANTES ET PARAMÈTRES GLOBAUX
# ==========================================
HBAR_MEV_S = 6.582119569e-13  # meV·s
A_TO_M = 1e-10                # 1 Å = 1e-10 m
N_sites = 1024                # Taille de la chaîne

# ==========================================
# 2. FONCTION DE SIMULATION COMPLÈTE
# ==========================================
def simuler_paquet(J=1.0, S=0.5, a=1.0, t_sim=1.5e-12, N_pas=50, marge=30):
    x = np.arange(N_sites) * a
    k_modes = 2 * np.pi * np.fft.fftfreq(N_sites, d=a)
    omega_k = (4 * J * S / HBAR_MEV_S) * np.sin(k_modes * a / 2.0)**2
    
    # --- Énergie théorique (meV) de chaque mode, utilisée pour <E> via psi_k
    E_k_mode = HBAR_MEV_S * omega_k  # = 4*J*S*sin^2(ka/2)
    
    # --- Préparation du paquet initial
    q0 = np.pi / (4.0 * a)
    sigma_q = 0.04 * (np.pi / a)
    x0 = N_sites * a / 4.0
    sigma_x0 = 1.0 / (2.0 * sigma_q)
    psi_0 = np.exp(-((x - x0)**2) / (4.0 * sigma_x0**2)) * np.exp(1j * q0 * x)
    psi_0 /= np.sqrt(np.sum(np.abs(psi_0)**2))
    
    # --- Vitesse théorique moyenne du paquet
    psi_k_0 = np.fft.fft(psi_0)
    poids_k = np.abs(psi_k_0)**2
    poids_k /= np.sum(poids_k)
    v_k = (2 * J * S * a / HBAR_MEV_S) * np.sin(k_modes * a)
    v_paquet_theo = float(np.sum(poids_k * v_k))
    
    def courbure(q):
        return (2 * J * S * (a**2) / HBAR_MEV_S) * np.cos(q * a)
    omega_seconde = courbure(q0)
    
    # --- Propagation
    temps = np.linspace(0, t_sim, N_pas)
    normes, centroides, variances, energies = [], [], [], []
    proba_bords_max = 0.0
    psi_historique = {}
    
    for idx, t in enumerate(temps):
        psi_t = np.fft.ifft(psi_k_0 * np.exp(-1j * omega_k * t))
        
        densite = np.abs(psi_t)**2
        norme_t = np.sum(densite)
        cx_t = np.sum(x * densite) / norme_t
        var_t = np.sum((x**2) * densite) / norme_t - cx_t**2
        
        # Énergie calculée sur l'ÉTAT PROPAGÉ (audit A)
        psi_k_t = np.fft.fft(psi_t)
        poids_k_t = np.abs(psi_k_t)**2
        poids_k_t /= np.sum(poids_k_t)  # renormalisation
        energie_t = float(np.sum(poids_k_t * E_k_mode))
        
        # Surveillance dynamique de la périodicité (audit B)
        proba_bords = np.sum(densite[:marge]) + np.sum(densite[-marge:])
        proba_bords_max = max(proba_bords_max, proba_bords)
        
        normes.append(norme_t)
        centroides.append(cx_t)
        variances.append(var_t)
        energies.append(energie_t)
        
        if idx in [0, N_pas//2, N_pas-1]:
            psi_historique[t] = densite
    
    # --- Tests de validation
    assert np.max(np.abs(np.array(normes) - 1.0)) < 1e-12, "Échec norme"
    assert np.max(np.abs(np.array(energies) - energies[0])) < 1e-10, \
        f"Échec énergie (calculée sur psi_t) : {np.max(np.abs(np.array(energies)-energies[0]))}"
    assert proba_bords_max < 1e-10, \
        f"Échec périodicité : proba bords = {proba_bords_max}"
    
    pente, _ = np.polyfit(temps, centroides, 1)
    erreur_vitesse = np.abs(pente - v_paquet_theo) / v_paquet_theo
    assert erreur_vitesse < 0.01, f"Échec vitesse : {erreur_vitesse}"
    
    var_theo_finale = variances[0] + ((omega_seconde * t_sim)**2) / (4.0 * variances[0])
    erreur_etalement = np.abs(variances[-1] - var_theo_finale) / var_theo_finale
    assert erreur_etalement < 0.05, f"Échec GVD : {erreur_etalement}"
    
    return {
        'x': x, 'temps': temps, 'centroides': centroides,
        'variances': variances, 'psi_historique': psi_historique,
        'pente': pente, 'v_theo': v_paquet_theo,
        'omega_sec': omega_seconde,
        'erreurs': {'vitesse': erreur_vitesse, 'etalement': erreur_etalement},
        'proba_bords_max': proba_bords_max
    }

# ==========================================
# 3. EXÉCUTION POUR DIFFÉRENTS J
# ==========================================
J_LIST = [0.5, 1.0, 2.0]
resultats = {}

print("======================================================================")
print("V101 — PROPAGATION DYNAMIQUE DU PAQUET DE MAGNONS (AUDIT RENFORCÉ)")
print("======================================================================")
for J in J_LIST:
    res = simuler_paquet(J=J)
    resultats[J] = res
    print(f"J = {J:.1f} meV | v_theo = {res['v_theo']*A_TO_M:6.2f} m/s | "
          f"v_mes = {res['pente']*A_TO_M:6.2f} m/s | "
          f"err_vit = {res['erreurs']['vitesse']:.2e} | "
          f"err_GVD = {res['erreurs']['etalement']:.2e} | "
          f"proba_bords = {res['proba_bords_max']:.2e}")
print("======================================================================")

# ==========================================
# 4. VISUALISATION MULTI-PANNEAUX (POUR J=1.0)
# ==========================================
res = resultats[1.0]
x = res['x']
centroides = np.array(res['centroides'])
variances = np.array(res['variances'])
temps = res['temps']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panneau 1 : Densité à différents instants
for t, densite in res['psi_historique'].items():
    axes[0].plot(x, densite, label=f"t = {t*1e12:.1f} ps")
axes[0].set_xlim(x[0] + 200, x[-1] - 200)
axes[0].set_title(r"Densité du paquet \(|\psi(x,t)|^2\)")
axes[0].set_xlabel("Position x (Å)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Panneau 2 : Centroïde et régression
axes[1].plot(temps * 1e12, centroides, 'o', markersize=3, label="Données")
axes[1].plot(temps * 1e12, res['pente'] * temps + centroides[0], 'r--',
             label=f"Fit : {res['pente']*A_TO_M:.1f} m/s")
axes[1].set_title(r"Mouvement du centroïde \(\langle x(t) \rangle\)")
axes[1].set_xlabel("Temps (ps)")
axes[1].set_ylabel("Position (Å)")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Panneau 3 : Variance (GVD)
var_theo = variances[0] + ((res['omega_sec'] * temps)**2) / (4 * variances[0])
axes[2].plot(temps * 1e12, variances, 'o', markersize=3, label="Données")
axes[2].plot(temps * 1e12, var_theo, 'k--', label="Théorie GVD")
axes[2].set_title(r"Étalement du paquet \(\sigma_x^2(t)\)")
axes[2].set_xlabel("Temps (ps)")
axes[2].set_ylabel("Variance (Å²)")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("v101_magnon_packet.png", dpi=150)
print("Figure sauvegardée : v101_magnon_packet.png")
