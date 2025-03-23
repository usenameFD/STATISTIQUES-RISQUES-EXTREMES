# Gestion des Risques Extrêmes avec Dash

Ce projet présente une implémentation Dash pour la gestion des risques extrêmes sur les actions et indices cotés. Un accent particulier est mis sur la calibration de la VaR et le backtesting.

## Fonctionnalités principales

### Préparation des données
- Importation des données financières (Date, Cours de clôture)
- Nettoyage et gestion des valeurs manquantes
- Calcul des log-rendements
- Visualisation des séries temporelles

### Estimation de la VaR
- **VaR Historique** : Méthode empirique et bootstrap avec intervalle de confiance
- **VaR Gaussienne** : Estimation paramétrique avec validation ex-ante (QQ-plots, scaling, diffusion d’actifs)
- **VaR Skew-Student** : Calibration par maximum de vraisemblance et comparaison avec la loi normale
- **VaR basée sur les valeurs extrêmes (TVE)** : Approches Maxima par bloc (MB) et Peak Over Threshold (PoT)

### Expected Shortfall (ES)
- Estimation empirique et théorique selon les différentes méthodes de VaR

### Protocole de Backtesting
- Recalibrage dynamique des modèles
- Évaluation des exceptions sur la période de test
- Indicateurs de nécessité de recalibrage

## Interface Dash

Une interface interactive permet :
- La sélection des périodes d’apprentissage et de test
- La calibration dynamique de la VaR
- L’exécution du backtesting avec affichage des résultats en temps réel
- La visualisation des distributions et diagnostics (QQ-plots, densité)

### Lancer l'application
```bash
python app.py
