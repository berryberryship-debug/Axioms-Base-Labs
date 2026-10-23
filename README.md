# Axioms Base Labs : Moteur de Calcul Solitonique et Quantique

**Auteur :** Xioranne (Xio) / Kush
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
*   **`validation.py`** : Suite de tests unitaires indépendants validant la convergence du noyau (phase globale, tolérances, unités).

### L'Orchestrateur (`master_pipeline.py`)
Le script racine `master_pipeline.py` fait le pont entre tous les modules. 
*   **Calibration :** Calibration de l'aire de l'impulsion pour une porte X (rotation de \(\pi\)).
*   **Simulation de référence :** Évolution sans bruit avec tolérances strictes (`rtol=1e-10`, `atol=1e-12`).
*   **Étude Monte-Carlo :** Injection de bruit 1/f multiplicatif sur l'amplitude de contrôle. Évaluation de la robustesse statistique sur 100 réalisations indépendantes.

## 3. Exécution du Pipeline

### Installation
```bash
pip install numpy scipy
