"""
V99 — Skyrmion 2D : audit topologique multi-méthodes
Axioms-QuantumLab

Méthodes :
    1. Différences finies (FD)
    2. Berg-Luscher (BL), angles solides sphériques
    3. Comparaison multi-résolution
    4. Estimation de l'ordre de convergence

Texture :
    Skyrmion de Neel axisymétrique
    Polarité centrale : -z
    Polarité périphérique : +z
    Vorticité : +1

Convention :
    Q = (1 / 4π) integral n · (∂x n × ∂y n) dx dy

Références :
    - Berg, B., & Lüscher, M. (1981). "Definition and statistical distributions of a topological number in the lattice O(3) sigma-model." Nuclear Physics B, 190(2), 412-424. (Méthode exacte Berg-Lüscher).
    - Dzyaloshinskii, I. E. (1958). "A thermodynamic theory of 'weak' ferromagnetism of antiferromagnetics." Journal of Physics and Chemistry of Solids, 4(4), 241-255. (Mécanisme physique sous-jacent).
    - Moriya, T. (1960). "Anisotropic superexchange interaction and weak ferromagnetism." Physical Review, 120(1), 91. (Formalisation de l'interaction DMI).
    - Nagaosa, N., & Tokura, Y. (2013). "Topological properties and dynamics of magnetic skyrmions." Nature Nanotechnology, 8(12), 899-911. (Revue générale sur la topologie des skyrmions).

Limites :
    - Domaine carré fini et ouvert.
    - Pas de minimisation énergétique.
    - Pas de dynamique temporelle.
    - La charge BL sur un domaine ouvert n'est pas nécessairement un entier exact.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv


# ============================================================
# 1. PARAMÈTRES
# ============================================================

L = 30.0
R_SK = 6.0
WALL_WIDTH = 1.5

POLARITY = 1
CHIRALITY = 1

RESOLUTIONS = [101, 201, 301, 601]
N_DISPLAY = 301

OUTPUT_DIR = Path("results_v99")
OUTPUT_DIR.mkdir(exist_ok=True)

# Convention de signe :
# POLARITY = +1 : cœur -z, périphérie +z
# POLARITY = -1 : cœur +z, périphérie -z


# ============================================================
# 2. PROFIL RADIAL RÉGULARISÉ
# ============================================================

def profil_radial(r):
    """
    Profil radial régulier au centre.

    tan(theta/2) = exp((R_SK-r)/WALL_WIDTH) * R_SK/r

    Ainsi :
        theta(0) = pi
        theta(R_SK) = pi/2
        theta(inf) -> 0

    Le facteur R_SK/r garantit sin(theta) ~ r au centre,
    ce qui régularise la dépendance azimutale pour m = ±1.
    """

    r = np.asarray(r, dtype=float)

    log_tan = (
        (R_SK - r) / WALL_WIDTH
        + np.log(R_SK / np.maximum(r, 1e-300))
    )

    # Évaluation stable de 2 arctan(exp(log_tan))
    theta = 2.0 * np.arctan(
        np.exp(np.clip(log_tan, -350.0, 350.0))
    )

    theta = np.where(r == 0.0, np.pi, theta)

    if POLARITY == -1:
        theta = np.pi - theta

    return theta


# ============================================================
# 3. CONSTRUCTION DU CHAMP DE SPINS
# ============================================================

def construire_skyrmion(N):
    x = np.linspace(-L / 2.0, L / 2.0, N)
    y = np.linspace(-L / 2.0, L / 2.0, N)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="xy")

    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)

    theta = profil_radial(R)
    phi = CHIRALITY * PHI

    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)

    spins = np.stack((nx, ny, nz), axis=-1)

    # Normalisation explicite
    normes = np.linalg.norm(spins, axis=-1, keepdims=True)
    spins /= np.maximum(normes, 1e-15)

    return x, y, X, Y, spins, dx, dy


# ============================================================
# 4. MÉTHODE 1 : DIFFÉRENCES FINIES
# ============================================================

def charge_differences_finies(spins, dx, dy):
    """
    Q_FD = integral q(x,y) dx dy

    q = n · (∂x n × ∂y n) / (4π)

    Différences centrées d'ordre 2 à l'intérieur.
    """

    dn_dy, dn_dx = np.gradient(
        spins,
        dy,
        dx,
        axis=(0, 1),
        edge_order=2
    )

    produit = np.cross(dn_dx, dn_dy)

    densite = np.sum(spins * produit, axis=-1) / (4.0 * np.pi)

    Q_fd = np.sum(densite) * dx * dy

    return float(Q_fd), densite


# ============================================================
# 5. MÉTHODE 2 : BERG-LÜSCHER
# ============================================================

def angle_solide(a, b, c):
    """
    Angle solide signé d'un triangle sphérique.

    Omega = 2 atan2(
        a · (b × c),
        1 + a·b + b·c + c·a
    )
    """

    numerateur = np.sum(
        a * np.cross(b, c),
        axis=-1
    )

    denominateur = (
        1.0
        + np.sum(a * b, axis=-1)
        + np.sum(b * c, axis=-1)
        + np.sum(c * a, axis=-1)
    )

    return 2.0 * np.arctan2(
        numerateur,
        denominateur
    )


def charge_berg_luscher(spins):
    """
    Triangulation orientée de chaque cellule carrée.

    Les axes sont :
        axe 0 : y croissant
        axe 1 : x croissant

    Les deux triangles sont orientés dans le sens
    positif du plan (x,y).
    """

    s00 = spins[:-1, :-1]
    s10 = spins[:-1, 1:]
    s11 = spins[1:, 1:]
    s01 = spins[1:, :-1]

    omega1 = angle_solide(s00, s10, s11)
    omega2 = angle_solide(s00, s11, s01)

    Q_bl = np.sum(omega1 + omega2) / (4.0 * np.pi)

    return float(Q_bl)


# ============================================================
# 6. CONTRÔLES DE LA TEXTURE
# ============================================================

def verifier_texture(spins, dx):
    normes = np.linalg.norm(spins, axis=-1)

    erreur_norme = float(
        np.max(np.abs(normes - 1.0))
    )

    assert np.all(np.isfinite(spins))
    assert erreur_norme < 1e-12

    assert dx < WALL_WIDTH / 3.0, (
        "Résolution insuffisante pour la largeur de paroi."
    )

    return erreur_norme


def controle_frontiere(spins):
    """
    Mesure l'écart de la frontière à la polarisation +z.

    La frontière du domaine carré est constituée
    des quatre côtés du tableau.
    """

    bord = np.concatenate([
        spins[0, :, :],
        spins[-1, :, :],
        spins[1:-1, 0, :],
        spins[1:-1, -1, :]
    ], axis=0)

    nz = bord[:, 2]

    deviation_nz = float(np.max(np.abs(nz - 1.0)))

    composante_transverse = np.sqrt(
        bord[:, 0]**2 + bord[:, 1]**2
    )

    max_transverse = float(
        np.max(composante_transverse)
    )

    return deviation_nz, max_transverse


# ============================================================
# 7. AUDIT MULTI-RÉSOLUTION
# ============================================================

resultats = []

print("=" * 76)
print("V99 — AUDIT TOPOLOGIQUE MULTI-RÉSOLUTION")
print("=" * 76)

for N in RESOLUTIONS:

    x, y, X, Y, spins, dx, dy = construire_skyrmion(N)

    erreur_norme = verifier_texture(spins, dx)

    Q_fd, densite = charge_differences_finies(
        spins, dx, dy
    )

    Q_bl = charge_berg_luscher(spins)

    ecart = abs(Q_fd - Q_bl)

    deviation_bord, transverse_bord = controle_frontiere(
        spins
    )

    resultats.append({
        "N": N,
        "h": dx,
        "Q_FD": Q_fd,
        "Q_BL": Q_bl,
        "ecart": ecart,
        "erreur_norme": erreur_norme,
        "deviation_bord": deviation_bord,
        "transverse_bord": transverse_bord
    })

    print(
        f"N={N:4d} | "
        f"h={dx:.6f} | "
        f"Q_FD={Q_fd:.9f} | "
        f"Q_BL={Q_bl:.9f} | "
        f"écart={ecart:.3e}"
    )


# ============================================================
# 8. ORDRE DE CONVERGENCE OBSERVÉ
# ============================================================

print("\n" + "=" * 76)
print("ORDRE DE CONVERGENCE")
print("=" * 76)

for i in range(1, len(resultats)):

    precedent = resultats[i - 1]
    courant = resultats[i]

    e1 = precedent["ecart"]
    e2 = courant["ecart"]

    h1 = precedent["h"]
    h2 = courant["h"]

    if e1 > 0.0 and e2 > 0.0 and h1 != h2:
        p = np.log(e1 / e2) / np.log(h1 / h2)
    else:
        p = np.nan

    courant["ordre_p"] = float(p)

    print(
        f"{precedent['N']:4d} -> {courant['N']:4d} | "
        f"p = {p:.5f}"
    )

resultats[0]["ordre_p"] = np.nan


# ============================================================
# 9. RÉFÉRENCE ANALYTIQUE
# ============================================================

"""
Pour le profil régularisé :

theta(0) = pi
theta(inf) = 0

Avec la convention :
Q = (m/2) [cos(theta(0)) - cos(theta(inf))]

et m = CHIRALITY.

Donc :
Q_infini = -CHIRALITY

Pour le domaine carré fini, la charge numérique peut
différer légèrement de cette limite à cause de la
polarisation périphérique non parfaitement uniforme.
"""

Q_analytique = -float(CHIRALITY)

print("\n" + "=" * 76)
print("RÉFÉRENCE ANALYTIQUE")
print("=" * 76)
print(f"Charge topologique idéale : {Q_analytique:.8f}")

dernier = resultats[-1]

print(f"Q_FD à N={dernier['N']}          : {dernier['Q_FD']:.9f}")
print(f"Q_BL à N={dernier['N']}          : {dernier['Q_BL']:.9f}")
print(f"Écart FD / BL              : {dernier['ecart']:.3e}")


# ============================================================
# 10. SAUVEGARDE CSV
# ============================================================

csv_path = OUTPUT_DIR / "v99_convergence.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    champs = [
        "N",
        "h",
        "Q_FD",
        "Q_BL",
        "ecart",
        "ordre_p",
        "erreur_norme",
        "deviation_bord",
        "transverse_bord"
    ]

    writer = csv.DictWriter(f, fieldnames=champs)
    writer.writeheader()

    for ligne in resultats:
        writer.writerow(ligne)

print(f"\nCSV sauvegardé : {csv_path}")


# ============================================================
# 11. VISUALISATION À LA RÉSOLUTION DE RÉFÉRENCE
# ============================================================

x, y, X, Y, spins, dx, dy = construire_skyrmion(N_DISPLAY)

Q_fd, densite = charge_differences_finies(spins, dx, dy)
Q_bl = charge_berg_luscher(spins)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Composante n_z
im0 = axes[0].imshow(
    spins[:, :, 2],
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1
)

axes[0].set_title(r"Composante \(n_z\)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_aspect("equal")
plt.colorbar(im0, ax=axes[0])

# Champ vectoriel
stride = max(1, N_DISPLAY // 25)

axes[1].quiver(
    X[::stride, ::stride],
    Y[::stride, ::stride],
    spins[::stride, ::stride, 0],
    spins[::stride, ::stride, 1],
    spins[::stride, ::stride, 2],
    cmap="coolwarm",
    pivot="mid"
)

axes[1].set_title("Texture vectorielle")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect("equal")

# Densité topologique
lim = np.max(np.abs(densite))

im2 = axes[2].imshow(
    densite,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="RdBu_r",
    vmin=-lim,
    vmax=lim
)

axes[2].set_title(r"Densité topologique \(q(x,y)\)")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_aspect("equal")
plt.colorbar(im2, ax=axes[2])

fig.suptitle(
    f"V99 — Skyrmion 2D | Q_FD={Q_fd:.6f} | Q_BL={Q_bl:.6f}"
)

plt.tight_layout()

figure_path = OUTPUT_DIR / "v99_skyrmion_2d.png"
plt.savefig(figure_path, dpi=180)
plt.close()


# ============================================================
# 12. EXPORT DES DONNÉES
# ============================================================

np.savez_compressed(
    OUTPUT_DIR / "v99_skyrmion_data.npz",
    x=x,
    y=y,
    spins=spins,
    densite_topologique=densite,
    Q_FD=Q_fd,
    Q_BL=Q_bl,
    Q_analytique=Q_analytique
)

print(f"Figure sauvegardée : {figure_path}")
print(f"Données sauvegardées : {OUTPUT_DIR / 'v99_skyrmion_data.npz'}")


# ============================================================
# 13. SYNTHÈSE
# ============================================================

print("\n" + "=" * 76)
print("SYNTHÈSE V99")
print("=" * 76)

print(f"Charge analytique idéale : {Q_analytique:.8f}")
print(f"Charge FD                : {Q_fd:.8f}")
print(f"Charge Berg-Lüscher      : {Q_bl:.8f}")
print(f"Écart FD / BL            : {abs(Q_fd-Q_bl):.3e}")

print(
    f"Erreur de normalisation : "
    f"{verifier_texture(spins, dx):.3e}"
)

print("Calcul terminé.")
print("=" * 76)
