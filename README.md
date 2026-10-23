# Axioms Base Labs : Moteur de Calcul Solitonique et Quantique

**Auteur :** Xioranne Lessard
**Cadre :** Architecture logicielle quantique et topologique unifiée
**Version :** 2.0.0

## 1. Vision : L'Ordre Implicite du Contrôle Quantique
Le paradigme actuel du contrôle quantique optimise la fidélité des portes logiques de manière isolée, considérant le bruit comme un ennemi statistique. Axioms Base Labs propose un renversement phénoménologique : la robustesse de l'information quantique ne s'obtient pas en luttant contre l'environnement, mais en la structurant géométriquement sous forme solitonique. 

Ce projet matérialise l'intersection exacte entre la dynamique de Hilbert (discrète) et la topologie non-linéaire (continue), avec un écosystème d'agents cognitifs autopoïétiques.

## 2. Architecture des Moteurs de Calcul

Le système repose sur des briques computationnelles souveraines, exécutables localement et verrouillées contre la décohérence numérique.

### Le Noyau Quantique (`core/`)
*   **`quantum_core.py`** : Noyau dynamique quantique. Résout l'équation de Schrödinger via `solve_ivp`. Inclut la validation stricte de l'hermiticité.
*   **`pulses.py`** : Génération d'impulsions (Gaussienne, DRAG, Sinusoïdale, Composites).
*   **`noise.py`** : Générateur de bruit coloré 1/f (bruit rose).
*   **`metrics.py`** : Métrologie quantique (Fidélité, Infidélité).
*   **`moire.py`** : Champ topologique Moiré avec stabilité numérique.
*   **`validation.py`** : Suite de tests unitaires validant la convergence du noyau.

### Les Expériences (Racine du projet)
*   **`master_pipeline.py`** : Le chef d'orchestre de base. Simule une porte X et fait un Monte-Carlo sur 100 réalisations pour tester la robustesse au bruit.
*   **`test.py`** : Test de validation rapide du moteur quantique (vérification de la norme et des états).
*   **`time_reversal_witness.py`** : Première implémentation du témoin d'inversion temporelle (modèle simple à 2 niveaux). Montre la sensibilité de la phase au profil temporel du bruit.
*   **`time_reversal_witness_tls.py`** : Modèle avancé (8x8). Ajoute un défaut matériel (TLS) et la dissipation via l'équation de Lindblad. Calcule la fidélité de Bell avec une correction hermitienne stricte.
*   **`scan_multi_tls.py`** : Scan automatique (le "Gargantuesque"). Balaye le nombre de TLS (1 et 2) et la force de couplage pour générer des cartes de chaleur montrant l'impact des défauts sur la cohérence quantique.
*   **`visualisation.py`** : Module de génération des figures scientifiques (Impulsion, Évolution, Monte-Carlo).

## 3. Validation et Preuves Numériques (Distinction Code/Preuve)

Les propriétés annoncées sont vérifiées par des tests automatisés. Voici la distinction entre ce que le code implémente et ce que les tests démontrent :

*   **Stabilité numérique et conservation de la norme** : 
    *   *Implémentation :* L'intégrateur `solve_ivp` avec tolérances strictes.
    *   *Preuve :* Le test `test_analytic_solution` vérifie que l'erreur de norme reste inférieure à `1e-7`.
*   **Validation de l'hermiticité des Hamiltoniens** : 
    *   *Implémentation :* Fonction `validate_hamiltonian` et `validate_static_operators`.
    *   *Preuve :* Les tests prouvent que les matrices non-hermitiennes sont rejetées (erreur fatale).
*   **Robustesse au bruit (Monte-Carlo)** : 
    *   *Implémentation :* Boucle sur 100 itérations avec bruit 1/f et RF.
    *   *Preuve :* La simulation affiche l'écart-type de la fidélité, démontrant statistiquement l'impact du bruit sur la porte quantique.

## 4. Exécution du Pipeline

### Installation
```bash
pip install numpy scipy matplotlib tqdm
