# Axioms Base Labs : Moteur de Calcul Solitonique et Quantique

**Auteur :** Xioranne Lessard
**Cadre :** Architecture logicielle quantique et topologique unifiée
**Version :** 2.3.0

## 1. Vision : L'Ordre Implicite du Contrôle Quantique
Le paradigme actuel du contrôle quantique optimise la fidélité des portes logiques de manière isolée, considérant le bruit comme un ennemi statistique. Axioms Base Labs propose un renversement phénoménologique : la robustesse de l'information quantique ne s'obtient pas en luttant contre l'environnement, mais en la structurant géométriquement sous forme solitonique. 

Ce projet matérialise l'intersection entre la dynamique de Hilbert (discrète, systèmes ouverts) et la topologie non-linéaire (continue, dispersion des ondes de spin), avec un écosystème d'agents cognitifs autopoïétiques.

## 2. Architecture des Moteurs de Calcul

Le système repose sur des briques computationnelles souveraines, exécutables localement et verrouillées contre la décohérence numérique.

### 2.1 Le Noyau Quantique (`core/`)
*   **`quantum_core.py`** : Noyau dynamique quantique. Résout l'équation de Schrödinger via `solve_ivp`. Inclut la validation stricte de l'hermiticité.
*   **`pulses.py`** : Génération d'impulsions (Gaussienne, DRAG, Sinusoïdale, Composites).
*   **`noise.py`** : Générateur de bruit coloré 1/f (bruit rose) et de bruit RF.
*   **`metrics.py`** : Métrologie quantique (Fidélité, Infidélité).
*   **`moire.py`** : Champ topologique Moiré avec stabilité numérique.
*   **`validation.py`** : Suite de tests unitaires validant la convergence du noyau.

### 2.2 Les Expériences Quantiques (Racine)
*   **`master_pipeline.py`** : Le chef d'orchestre de base. Simule une porte X et fait un Monte-Carlo sur 100 réalisations pour tester la robustesse au bruit.
*   **`time_reversal_witness_tls.py`** : Modèle avancé (8x8). Ajoute un défaut matériel (TLS) et la dissipation via l'équation de Lindblad. Calcule la fidélité de Bell avec une correction hermitienne stricte et un contrôle de réversibilité unitaire. Implémente deux méthodes de résolution numérique (voir section 3.1).
*   **`scan_multi_tls.py`** : Scan automatique. Balaye le nombre de TLS (1 et 2) et la force de couplage pour générer des cartes de chaleur montrant l'impact des défauts sur la cohérence quantique.
*   **`visualisation.py`** : Module de génération des figures scientifiques.

### 2.3 Module Matière Condensée & Magnétisme (V99 - V102)
*   **`v99_skyrmion_2d.py`** : Audit topologique d'un skyrmion de Néel 2D. Calcule la charge topologique via deux méthodes (différences finies et angles solides de Berg-Lüscher).
*   **`v100_magnon_dispersion.py`** : Relation de dispersion des magnons 1D (Heisenberg ferromagnétique), calcul de la vitesse physique et de la masse effective en unités SI.
*   **`v101_magnon_packet.py`** : Propagation dynamique spatio-temporelle d'un paquet d'ondes de spin via FFT, avec tests stricts d'unitarité et d'étalement (GVD).
*   **`v102_magnon_dmi.py`** : Effet Dzyaloshinskii-Moriya (DMI), instabilité ferromagnétique (mode mou), stabilisation par champ externe et dispersion autour d'un fondamental spiralé.

## 3. Validation et Preuves Numériques

Les propriétés annoncées sont vérifiées par des tests automatisés. Voici la distinction entre ce que le code implémente et ce que les tests démontrent :

*   **Stabilité numérique et conservation de la norme** : 
    *   *Implémentation :* L'intégrateur `solve_ivp` avec tolérances strictes (et FFT unitaire pour les magnons).
    *   *Preuve :* Le test `test_analytic_solution` vérifie que l'erreur de norme reste inférieure à `1e-7` (quantique) et `1e-12` (magnons V101).
*   **Validation de l'hermiticité des Hamiltoniens** : 
    *   *Implémentation :* Fonction `validate_hamiltonian` et `validate_static_operators`.
    *   *Preuve :* Les tests prouvent que les matrices non-hermitiennes sont rejetées (erreur fatale).
*   **Robustesse au bruit (Monte-Carlo)** : 
    *   *Implémentation :* Boucle sur plusieurs itérations avec bruit 1/f et RF.
    *   *Preuve :* La simulation affiche l'écart-type de la fidélité, démontrant statistiquement l'impact du bruit sur la porte quantique.

### 3.1 Approches Numériques pour la Dissipation (Lindblad)
Le module `time_reversal_witness_tls.py` implémente deux formalismes distincts pour valider la cohérence des calculs :
1.  **Méthode Directe (8x8)** : Calcul des commutateurs avec des matrices de densité 8x8. Privilégiée pour la génération des résultats principaux en raison de sa rapidité d'exécution.
2.  **Méthode de Liouville (64x64)** : Vectorisation de la matrice densité et construction d'un super-opérateur. Sert de validation croisée.

| Chirp (`kappa`) | `A_control` (Directe) | `A_control` (Liouville) |
| :--- | :--- | :--- |
| 0.000 | `~ 2.14e-06` | `~ -3.47e-06` |
| 0.020 | `-1.1012e-02` | `-1.1012e-02` |
| 0.050 | `-2.7337e-02` | `-2.7336e-02` |
| 0.100 | `-5.3308e-02` | `-5.3308e-02` |

*Les deux méthodes numériques convergent vers les mêmes résultats physiques, validant ainsi la robustesse de l'algorithme et l'absence d'erreur de vectorisation.*

## 4. Résultats Scientifiques

### 4.1 Résultats Quantiques (TLS avec Lindblad)
*   **Fidélité de l'état de Bell** : Chute de 99.9% (sans dissipation) à ~91% (avec dissipation et bruit).
*   **Concurrence (Intrication)** : Démontre la destruction de l'intrication par la dissipation (passage de 1.0 à 0.13).
*   **Witness d'inversion temporelle** : Reste robuste (autour de 0.95 - 1.0) malgré la présence de bruit.

### 4.2 Résultats Magnétiques (Matière Condensée)

**V99 — Champ de spins 2D et audit topologique multi-méthodes**
**[Modèle et méthode]** Construction d'un skyrmion de Néel axisymétrique sur un domaine carré fini, avec un profil radial régularisé (évitant la singularité en r=0). Le calcul de la charge topologique \(Q\) est effectué selon deux méthodes indépendantes : Différences finies (FD) d'ordre 2, et méthode exacte de Berg-Lüscher (BL) basée sur les angles solides sphériques.
**[Vérifié numériquement]** Convergence croisée des deux méthodes (l'écart tend vers 0 avec N). L'étude multi-résolution (N=101 à 601) estime l'ordre de convergence \(p\) de la méthode FD. Un contrôle de bord vérifie la polarisation \(+z\).
**[Limite]** Domaine carré fini et ouvert. Pas de minimisation énergétique ni de dynamique temporelle. La charge BL n'est pas nécessairement un entier exact sur un domaine ouvert.
**[Références V99]** 
- Berg, B., & Lüscher, M. (1981). *Definition and statistical distributions of a topological number in the lattice O(3) sigma-model*. Nuclear Physics B.
- Nagaosa, N., & Tokura, Y. (2013). *Topological properties and dynamics of magnetic skyrmions*. Nature Nanotechnology.

**V100 — Dispersion des Magnons 1D**
**[Démontré analytiquement]** Théorie linéaire des ondes de spin. Calcul de l'énergie \(E(q)\) et de la vitesse physique.
**[Vérifié numériquement]** La masse effective est calculée en unités SI. La dispersion est quadratique près de \(\Gamma\) (contrairement aux phonons).

**V101 — Dynamique et Propagation**
**[Vérifié numériquement]** Propagation d'un paquet d'ondes par FFT avec validation stricte de l'unitarité, de la conservation de l'énergie et de l'étalement (GVD). Le centroïde reste bien confiné dans la boîte de simulation.

**V102 — Effet Dzyaloshinskii-Moriya (DMI)**
**[Démontré analytiquement]** La DMI brise la symétrie d'inversion. L'état ferromagnétique colinéaire devient instable (mode mou) si le champ externe ne compense pas la chiralité. 
**[Observé numériquement]** Les trois régimes sont tracés : instabilité sans champ, stabilisation avec champ critique \(h_c\), et formation spontanée d'une spirale (hélicoïde) avec un mode de Goldstone sans gap à \(q=0\) (\(E(q) = 2S\sqrt{J^2+D^2}|\sin(qa/2)|\)).
**[Références V102]**
- Dzyaloshinskii, I. E. (1958). *A thermodynamic theory of "weak" ferromagnetism of antiferromagnetics*. Journal of Physics and Chemistry of Solids.
- Moriya, T. (1960). *Anisotropic superexchange interaction and weak ferromagnetism*. Physical Review.

## 5. Exécution du Pipeline

### Installation et lancement
```bash
pip install numpy scipy matplotlib
python master_pipeline.py
python scan_multi_tls.py
