# Axioms Base Labs : Moteur de Calcul Solitonique et Quantique

**Auteur :** Xioranne Lessard
**Cadre :** Architecture logicielle quantique et topologique unifiée
**Version :** 2.2.0

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
*   **`time_reversal_witness_tls.py`** : Modèle avancé (8x8). Ajoute un défaut matériel (TLS) et la dissipation via l'équation de Lindblad. Calcule la fidélité de Bell avec une correction hermitienne stricte et un contrôle de réversibilité unitaire. 
*   **`scan_multi_tls.py`** : Scan automatique. Balaye le nombre de TLS (1 et 2) et la force de couplage pour générer des cartes de chaleur montrant l'impact des défauts sur la cohérence quantique.
*   **`visualisation.py`** : Module de génération des figures scientifiques.

### 2.3 Module Matière Condensée & Magnétisme (V99 - V102)
*   **`v99_skyrmion_2d.py`** : Génération d'un champ de spins 2D (skyrmion) et calcul de la charge topologique.
*   **`v100_magnon_dispersion.py`** : Relation de dispersion des magnons 1D (Heisenberg ferromagnétique) et calcul de la masse effective.
*   **`v101_magnon_packet.py`** : Propagation dynamique spatio-temporelle d'un paquet d'ondes de spin via FFT.
*   **`v102_magnon_dmi.py`** : Effet Dzyaloshinskii-Moriya (DMI) et transition vers un fondamental spiralé (mode mou et mode de Goldstone).

---

## 3. Validation et Preuves Numériques

Les propriétés annoncées sont vérifiées par des tests automatisés. Voici la distinction entre ce que le code implémente et ce que les tests démontrent :

*   **Stabilité numérique et conservation de la norme** : 
    *   *Implémentation :* L'intégrateur `solve_ivp` avec tolérances strictes (et FFT unitaire pour les magnons).
    *   *Preuve :* Le test `test_analytic_solution` vérifie que l'erreur de norme reste inférieure à `1e-7` (quantique) et `1e-12` (V101 magnons).
*   **Validation de l'hermiticité des Hamiltoniens** : 
    *   *Implémentation :* Fonction `validate_hamiltonian` et `validate_static_operators`.
    *   *Preuve :* Les tests prouvent que les matrices non-hermitiennes sont rejetées (erreur fatale).
*   **Robustesse au bruit (Monte-Carlo)** : 
    *   *Implémentation :* Boucle sur plusieurs itérations avec bruit 1/f et RF.

### 3.1 Approches Numériques pour la Dissipation (Lindblad)
Le module `time_reversal_witness_tls.py` implémente deux formalismes distincts pour valider la cohérence des calculs :
1.  **Méthode Directe (8x8)** : Calcul des commutateurs avec des matrices de densité 8x8. Privilégiée pour la génération des résultats principaux en raison de sa rapidité.
2.  **Méthode de Liouville (64x64)** : Vectorisation de la matrice densité et construction d'un super-opérateur. Sert de validation croisée.

| Chirp (`kappa`) | `A_control` (Directe) | `A_control` (Liouville) |
| :--- | :--- | :--- |
| 0.000 | `~ 2.14e-06` | `~ -3.47e-06` |
| 0.020 | `-1.1012e-02` | `-1.1012e-02` |
| 0.050 | `-2.7337e-02` | `-2.7336e-02` |
| 0.100 | `-5.3308e-02` | `-5.3308e-02` |

*Les deux méthodes convergent, validant l'absence d'erreur de vectorisation.*

---

## 4. Résultats Scientifiques

### 4.1 Résultats Quantiques (TLS avec Lindblad)
*   **Fidélité de l'état de Bell** : Chute de 99.9% (sans dissipation) à ~91% (avec dissipation et bruit).
*   **Concurrence (Intrication)** : Démontre la destruction de l'intrication par la dissipation (passage de 1.0 à 0.13).
*   **Witness d'inversion temporelle** : Reste robuste (autour de 0.95 - 1.0) malgré la présence de bruit.

### 4.2 Résultats Magnétiques (Matière Condensée)

**V100 — Dispersion des Magnons 1D**
**[Démontré analytiquement]** Théorie linéaire des ondes de spin. Calcul de l'énergie \(E(q)\) et de la vitesse physique.
**[Vérifié numériquement]** La masse effective est calculée en unités SI. La dispersion est quadratique près de \(\Gamma\) (contrairement aux phonons).

**V101 — Dynamique et Propagation**
**[Vérifié numériquement]** Propagation d'un paquet d'ondes par FFT avec validation stricte de l'unitarité, de la conservation de l'énergie et de l'étalement (GVD). Le centroïde reste bien confiné dans la boîte de simulation.

**V102 — Effet Dzyaloshinskii-Moriya (DMI)**
**[Démontré analytiquement]** La DMI brise la symétrie d'inversion. L'état ferromagnétique colinéaire devient instable (mode mou) si le champ externe ne compense pas la chiralité. 
**[Observé numériquement]** Les trois régimes sont tracés : instabilité sans champ, stabilisation avec champ critique \(h_c\), et formation spontanée d'une spirale (hélicoïde) avec un mode de Goldstone sans gap à \(q=0\) (\(E(q) = 2S\sqrt{J^2+D^2}|\sin(qa/2)|\)).

---

## 5. Exécution du Pipeline

### Installation et lancement
```bash
python v100_magnon_dispersion.py
python v101_magnon_packet.py
python v102_magnon_dmi.py
***

### 💡 Pourquoi cette version est meilleure :
1. **Plus de doublon** : Le paragraphe répété sur Liouville a été supprimé.
2. **Structure claire** : On voit immédiatement que le dépôt possède deux grandes familles de moteurs (le Noyau Quantique et le module Magnétisme).
3. **Cohérence épistémique** : J'ai ajouté les niveaux de preuve (`[Démontré analytiquement]`, `[Vérifié numériquement]`) pour la partie magnons, ce qui garde la même rigueur que ton texte sur le TLS.
4. **Mise à jour des versions** : J'ai passé la version à `2.2.0` pour marquer cette belle évolution.

Tu n'as plus qu'à copier-coller ce bloc dans ton `README.md` sur GitHub !
pip install numpy scipy matplotlib
python master_pipeline.py
python scan_multi_tls.py
