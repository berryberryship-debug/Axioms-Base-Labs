⁹# Axioms Base Labs : Moteur de Calcul Solitonique et Quantique

**Auteur :** Xioranne Lessard
**Cadre :** Architecture logicielle quantique et topologique unifiée
**Version :** 2.1.0

## 1. Vision : L'Ordre Implicite du Contrôle Quantique
Le paradigme actuel du contrôle quantique optimise la fidélité des portes logiques de manière isolée, considérant le bruit comme un ennemi statistique. Axioms Base Labs propose un renversement phénoménologique : la robustesse de l'information quantique ne s'obtient pas en luttant contre l'environnement, mais en la structurant géométriquement sous forme solitonique. 

Ce projet matérialise l'intersection exacte entre la dynamique de Hilbert (discrète) et la topologie non-linéaire (continue), avec un écosystème d'agents cognitifs autopoïétiques.

## 2. Architecture des Moteurs de Calcul

Le système repose sur des briques computationnelles souveraines, exécutables localement et verrouillées contre la décohérence numérique.

### Le Noyau Quantique (`core/`)
*   **`quantum_core.py`** : Noyau dynamique quantique. Résout l'équation de Schrödinger via `solve_ivp`. Inclut la validation stricte de l'hermiticité.
*   **`pulses.py`** : Génération d'impulsions (Gaussienne, DRAG, Sinusoïdale, Composites).
*   **`noise.py`** : Générateur de bruit coloré 1/f (bruit rose) et de bruit RF.
*   **`metrics.py`** : Métrologie quantique (Fidélité, Infidélité).
*   **`moire.py`** : Champ topologique Moiré avec stabilité numérique.
*   **`validation.py`** : Suite de tests unitaires validant la convergence du noyau.

### Les Expériences (Racine du projet)
*   **`master_pipeline.py`** : Le chef d'orchestre de base. Simule une porte X et fait un Monte-Carlo sur 100 réalisations pour tester la robustesse au bruit.
*   **`time_reversal_witness_tls.py`** : Modèle avancé (8x8). Ajoute un défaut matériel (TLS) et la dissipation via l'équation de Lindblad. **Optimisation mathématique : la dissipation est vectorisée dans l'espace de Liouville (super-opérateur 64x64), permettant une intégration plus rapide et plus stable.** Calcule la fidélité de Bell avec une correction hermitienne stricte et un contrôle de réversibilité unitaire.
*   **`scan_multi_tls.py`** : Scan automatique. Balaye le nombre de TLS (1 et 2) et la force de couplage pour générer des cartes de chaleur montrant l'impact des défauts sur la cohérence quantique.
*   **`visualisation.py`** : Module de génération des figures scientifiques.

## 3. Validation et Preuves Numériques

## 3.1 Comparaison des Méthodes Numériques (Validation Croisée)

Pour garantir la robustesse de nos résultats, l'évolution de la matrice densité a été implémentée selon deux formalismes mathématiques distincts. Cette approche permet de valider que la physique simulée ne dépend pas de l'algorithme choisi.

*   **Méthode A (Commutateurs Directs)** : Résolution de l'équation de Lindblad avec des matrices 8x8 via `solve_ivp`. Méthode classique, rapide, utilisée pour générer les figures principales.
*   **Méthode B (Espace de Liouville)** : Vectorisation de la matrice densité en un vecteur de taille 64, et construction du super-opérateur de Liouville (64x64) à chaque pas de temps.

**Résultats comparatifs pour le contrôle (kappa = 0.0, sans chirp) :**

| Métrique | Méthode A (Directe) | Méthode B (Liouville) | Interprétation |
| :--- | :--- | :--- | :--- |
| `A_control` (Moyenne) | `~ 2.14e-06` | `~ -3.48e-06` | Les deux méthodes donnent zéro (noyé dans l'erreur SEM), confirmant que sans chirp, il n'y a pas d'asymétrie temporelle. |
| `A_both` (Moyenne) | `~ -4.99e-05` | `~ -5.03e-05` | Les résultats sont du même ordre de grandeur (bruit statistique). |
| Erreur Round-Trip Unitaire | `3.19e-06` | `9.24e-06` | La méthode B accumule un peu plus d'erreur d'arrondi due aux matrices 64x64, mais reste sous la barre du `1e-5`, ce qui est excellent. |

**Conclusion** : Les deux méthodes convergent vers les mêmes conclusions physiques (asymétrie nulle à kappa=0). La méthode A est privilégiée pour la production de données (plus rapide), tandis que la méthode B sert de validation théorique avancée.
Les propriétés annoncées sont vérifiées par des tests automatisés. Voici la distinction entre ce que le code implémente et ce que les tests démontrent :

*   **Stabilité numérique et conservation de la norme** : 
    *   *Implémentation :* L'intégrateur `solve_ivp` avec tolérances strictes.
    *   *Preuve :* Le test `test_analytic_solution` vérifie que l'erreur de norme reste inférieure à `1e-7`.
*   **Validation de l'hermiticité des Hamiltoniens** : 
    *   *Implémentation :* Fonction `validate_hamiltonian` et `validate_static_operators`.
    *   *Preuve :* Les tests prouvent que les matrices non-hermitiennes sont rejetées (erreur fatale).
*   **Vectorisation de Liouville** : 
    *   *Implémentation :* Construction de la matrice de Liouville (64x64) avec `np.kron` et intégration d'un vecteur d'état de taille 64 (ordre Fortran).
    *   *Preuve :* Le code résout exactement la même physique, mais avec un gain d'efficacité computationnelle en évitant les produits matriciels redondants.
*   **Robustesse au bruit (Monte-Carlo)** : 
    *   *Implémentation :* Boucle sur plusieurs itérations avec bruit 1/f et RF.
    *   *Preuve :* La simulation affiche l'écart-type de la fidélité, démontrant statistiquement l'impact du bruit sur la porte quantique.

## 4. Résultats Scientifiques

Les simulations sur laptop (modèle TLS avec Lindblad) ont généré les résultats suivants :
*   **Fidélité de l'état de Bell** : Chute de 99.9% (sans dissipation) à ~91% (avec dissipation et bruit).
*   **Concurrence (Intrication)** : Démontre la destruction de l'intrication par la dissipation (passage de 1.0 à 0.13).
*   **Witness d'inversion temporelle** : Reste robuste (autour de 0.95 - 1.0) malgré la présence de bruit.

*(La figure `figure_principale.png` générée par le code illustre ces trois phénomènes).*

## 5. Exécution du Pipeline

### Installation et lancement
```bash
pip install numpy scipy matplotlib
python master_pipeline.py
python scan_multi_tls.py
