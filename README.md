# BankingAnalysisMlflow

### Create Conda environment

##### run below commands in terminal but make sure conda is installed or use anaconda prompt which you will get as part of anaconda installation

1. `conda create -n envname python=3.9 ipykernel`
it will create a conda env named envname and install python version 3.9 and a ipykernel inside this environment

2. Activate the environment
`conda activate envname`

3. add newly created environment to the notebook as kernel
`python -m ipykernel install --user --name=envname`

4. install notebook inside the environment
`pip install notebook`

5. Now install all required dependencies to run this notebook

* `pip install pandas`
* `pip install numpy`
* `pip install scikit-learn`
* `pip install imblearn`
* `pip install matplotlib`
* `pip install mlflow`

Now open the notebook using below command: (from the anaconda prompt inside conda environment)

`jupyter notebook`

BankingAnalysisMlflow/
│
├── data/
│   └── banking.csv               # Les données brutes à analyser.
│
├── notebooks/
│   └── analysis.ipynb         # Notebook pour analyse des données.
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py       # Importation, nettoyage, et préparation des données.
│   ├── exploratory_analysis.py   # Analyse univariée et multivariée.
│   ├── feature_engineering.py    # Encodage des variables catégorielles et autres transformations.
│   ├── model_evaluation.py       # Évaluation de la variable cible et division des données.
│   ├── model_selection.py        # Sélection et comparaison des modèles.
│   ├── hyperparameter_tuning.py  # Optimisation des hyperparamètres.
│   └── deployment.py             # Scripts pour le déploiement du modèle.
│
├── tests/
│   ├── test_data_preparation.py  # Tests pour la préparation des données.
│   └── test_models.py            # Tests pour les modèles.
│
├── requirements.txt              # Fichier contenant les dépendances.
├── .gitignore                    # Fichier pour ignorer les fichiers/dossiers non nécessaires.
└── README.md                     # Documentation sur le projet, l'installation, et l'exécution.

