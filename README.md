# BankingAnalysisMlflow


## Description

Ce projet vise à développer un système de prédiction de souscription à un service financier en utilisant des techniques de Machine Learning et de MLOps. Les données utilisées sont issues d'une campagne marketing précédente, incluant des informations sur les clients et leur décision finale de souscrire ou non. L'objectif principal est de construire un pipeline MLOps complet, incluant l'analyse exploratoire des données, le nettoyage, la construction et l'évaluation de modèles de prédiction, ainsi que le déploiement du modèle final en production.


## Étapes du projet

1. **Importation des données** : Les données seront importées à partir d'un fichier CSV contenant des informations sur les clients et leur décision de souscrire ou non.
2. **Analyse exploratoire des données** : Une analyse univariée et multivariée sera effectuée pour comprendre la distribution des données et les relations entre les variables.
3. **Prétraitement des données** : Les données seront nettoyées et prétraitées pour gérer les valeurs manquantes et aberrantes, ainsi que pour encoder les variables catégorielles.
4. **Construction de modèles** : Différents modèles de prédiction seront construits et évalués, tels que la régression logistique, les arbres de décision et les méthodes d'ensemble.
5. **Optimisation des hyperparamètres** : Les hyperparamètres des modèles seront optimisés pour améliorer leurs performances.
6. **Déploiement du modèle** : Le meilleur modèle sera déployé en production à l'aide de MLflow, avec une API pour permettre son utilisation dans des applications ou des services externes.

## Commencer

Pour exécuter cette application en local, suivez les étapes ci-dessous :

### Prérequis

#### Create Conda environment

##### run below commands in terminal but make sure conda is installed or use anaconda prompt which you will get as part of anaconda installation

1. `conda create -n envname python=3.9 ipykernel`
it will create a conda env named envname and install python version 3.9 and a ipykernel inside this environment

2. Activate the environment
`conda activate envname`

3. add newly created environment to the notebook as kernel
`python -m ipykernel install --user --name=envname`

4. install notebook inside the environment
`pip install notebook`

#### ## Installation

2. Installer les dépendances requises : `pip install -r requirements.txt`

* `pip install pandas`
* `pip install numpy`
* `pip install scikit-learn`
* `pip install imblearn`
* `pip install matplotlib`
* `pip install mlflow`

## Utilisation

1. Exécuter le notebook Jupyter `notebook.ipynb` pour suivre le processus de développement du modèle.
2. Pour déployer le modèle en production, suivre les instructions dans `deployement.md`.

## Contributeurs

- Alimou DIALLO (@santoudllo): Data engineer
- GBE Grâce (@keagnon): Data engineer
- Melissa ADIB (@melissa) : Data engineer


## Licence

Ce projet est sous licence MIT. N'hésitez pas à utiliser et modifier le code pour vos propres projets.
