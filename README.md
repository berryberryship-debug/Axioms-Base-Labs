# Axioms Base Labs : Moteur de Calcul Solitonique et Quantique

**Auteur :** Xioranne Lessard
**Cadre :** Architecture logicielle quantique et topologique unifiée
**Version :** 1.2.0

## 1. Vision : L'Ordre Implicite du Contrôle Quantique
Le paradigme actuel du contrôle quantique optimise la fidélité des portes logiques de manière isolée, considérant le bruit comme un ennemi statistique. Axioms Base Labs propose un renversement phénoménologique : la robustesse de l'information quantique ne s'obtient pas en luttant contre l'environnement, mais en la structurant géométriquement sous forme solitonique. 

Ce projet matérialise l'intersection exacte entre la dynamique de Hilbert (discrète) et la topologie non-linéaire (continue), avec un écosystème d'agents cognitifs autopoïétiques.

## 2. Architecture des Moteurs de Calcul

Le système repose sur des briques computationnelles souveraines, exécutables localement et verrouillées contre la décohérence numérique.

### Le Noyau Quantique (`core/`)
*   **`quantum_core.py`** : Noyau dynamique quantique. Résout l'équation de Schrödinger dépendante du temps via `scipy.integrate.solve_ivp`. Inclut la validation stricte de l'hermiticité des Hamiltoniens et le contrôle de la norme des états.
*   **`pulses.py`** : Génération d'impulsions de contrôle (Gaussienne tronquée, DRAG avec dérivée analytique exacte, Sinusoïdale, Composites).
*   **`noise.py`** : Générateur de bruit coloré 1/f (bruit rose) reproductible par générateur moderne PCG64 (`np.random.default_rng`).
*   **`metrics.py`** : Métrologie quantique. Calcul de la fidélité et de l'infidélité des états purs avec validation de normalisation.
*   **`moire.py`** : Champ topologique Moiré avec échantillonnage spatial. Implémente une stabilité numérique stricte (gestion des NaN/Inf et broadcasting).
*   **`validation.py`** : Suite de tests unitaires indépendants validant la convergence du noyau.

### L'Orchestrateur (`master_pipeline.py`)
Le script racine `master_pipeline.py` fait le pont entre tous les modules. 
*   **Calibration :** Calibration de l'aire de l'impulsion pour une porte X (rotation de \(\pi\)).
*   **Simulation de référence :** Évolution sans bruit avec tolérances strictes (`rtol=1e-10`, `atol=1e-12`).
*   **Étude Monte-Carlo :** Injection de bruit 1/f multiplicatif sur l'amplitude de contrôle. Évaluation de la robustesse statistique sur 100 réalisations indépendantes.

## 3. Validation et Preuves Numériques (Distinction Code/Preuve)

Les propriétés annoncées dans ce document sont vérifiées par des tests automatisés (le fichier `core/validation.py`). Voici la distinction entre ce que le code implémente et ce que les tests démontrent :

*   **Stabilité numérique et conservation de la norme :** 
    *   *Implémentation :* L'intégrateur `solve_ivp` avec tolérances strictes.
    *   *Preuve :* Le test `test_analytic_solution` vérifie que l'erreur de norme reste inférieure à `1e-7` sur toute la trajectoire.
*   **Validation de l'hermiticité des Hamiltoniens :** 
    *   *Implémentation :* Fonction `validate_hamiltonian`.
    *   *Preuve :* Le test `test_invalid_inputs` prouve que les matrices non-hermitiennes sont rejetées avec une `ValueError`.
*   **Robustesse au bruit (Monte-Carlo) :** 
    *   *Implémentation :* Boucle de 100 itérations avec bruit 1/f.
    *   *Preuve :* La simulation affiche l'écart-type de la fidélité, démontrant statistiquement l'impact du bruit sur la porte quantique.

## 4. Exécution du Pipeline

### Installation
```bash
## 5. Résultats Visuels

Voici les figures clés générées par les simulations :

### Impulsion de contrôle
![Impulsion](lien_vers_ton_image_01)

### Évolution du Qubit
![Évolution](lien_vers_ton_image_02)

### Distribution de la Fidélité
![Fidélité](lien_vers_ton_image_03)pip install numpy scipy
