import mlflow
from mlflow_utils import create_mlflow_experiment

if __name__=="__main__":

    experiment_id = create_mlflow_experiment(
        experiment_name="ProjetMLFLOW",
        artifact_location="testing_mlflow_artifact",
        tags={"env": "dev", "version": "1.0.0"},
    )
# print(f"Experiment ID: {experiment_id}")

client = MlflowClient()
with mlflow.start_run(run_name="first_run_rf") as run:
    mlflow.sklearn.log_model(artifact_path="rfr_model", sk_model=RandomForestRegressor(), registered_model_name="RandomForestRegressor")
    model_name = "RandomForest"
    # create registered model
    client.create_registered_model(model_name)
