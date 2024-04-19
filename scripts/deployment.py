import mlflow
from mlflow_utils import create_mlflow_experiment

# Définition de la fonction, assurez-vous qu'elle est dans votre script ou importée correctement
def create_mlflow_experiment(
    experiment_name="ProjetMLFLOW",
    artifact_location="testing_mlflow_artifact",
    tags={"env": "dev", "version": "1.0.0"}
) -> str:
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except Exception as e:
        print(f"Experiment {experiment_name} already exists: {str(e)}")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id

# Appel de la fonction sans passer d'arguments car les valeurs par défaut sont utilisées
# experiment_id = create_mlflow_experiment()
# print(f"Experiment ID: {experiment_id}")