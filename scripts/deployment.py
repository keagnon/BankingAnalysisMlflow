import mlflow
from mlflow import MlflowClient
from mlflow_utils import create_mlflow_experiment,get_mlflow_experiment
from analyse import *
from mlflow.types.schema import Schema
from mlflow.types.schema import ColSpec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

if __name__=="__main__":

     experiment_id = create_mlflow_experiment(
        experiment_name="ProjetMLFLOW",
        artifact_location="testing_mlflow_artifact",
        tags={"env": "dev", "version": "1.0.0"},
    )
# print(f"Experiment ID: {experiment_id}")

client = MlflowClient()
experiment = get_mlflow_experiment(experiment_name="ProjetMLFLOW")

# Random Forest Classifier
with mlflow.start_run(run_name="logging_models_rf_brute", experiment_id=221465795260439609) as run:
    mlflow.sklearn.autolog()
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train_upsampled, Y_train_upsampled)
    y_pred_rf = rfc.predict(X_test)

# Logistic Regression
with mlflow.start_run(run_name="logging_models_lr_brute", experiment_id=221465795260439609) as run:
    mlflow.sklearn.autolog()
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_upsampled, Y_train_upsampled)
    y_pred_lr = lr.predict(X_test)

# Gradient Boosting Classifier
with mlflow.start_run(run_name="logging_models_gb_brute", experiment_id=221465795260439609) as run:
    mlflow.sklearn.autolog()
    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc.fit(X_train_upsampled, Y_train_upsampled)
    y_pred_gb = gbc.predict(X_test)

