import mlflow
from mlflow import MlflowClient
from mlflow_utils import create_mlflow_experiment,get_mlflow_experiment
from analyse import *
from mlflow.types.schema import Schema
from mlflow.types.schema import ColSpec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

if __name__=="__main__":

    experiment_id = create_mlflow_experiment(
        experiment_name="ProjetMLFLOW",
        artifact_location="testing_mlflow_artifact",
        tags={"env": "dev", "version": "1.0.0"},
    )
# print(f"Experiment ID: {experiment_id}")

client = MlflowClient()
experiment_id=178385101586149999

# def optimize_model(model, param_grid, X_train, Y_train):
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
#     grid_search.fit(X_train, Y_train)
#     best_model = grid_search.best_estimator_
#     return best_model, grid_search.best_params_

# # Hyperparameters and models setup
# rf_params = {'n_estimators': [120, 300], 'max_depth': [8, 20, None]}
# lr_params = {'C': [0.1, 1, 11]}
# gb_params = {'n_estimators': [150, 250], 'learning_rate': [0.05, 0.1]}

# # Random Forest Classifier
# with mlflow.start_run(run_name="logging_models_rf", experiment_id=experiment_id):
#     mlflow.sklearn.autolog()
#     rf_model = RandomForestClassifier(random_state=42)
#     best_rf, best_rf_params = optimize_model(rf_model, rf_params, X_train_upsampled, Y_train_upsampled)
#     y_pred_rf = best_rf.predict(X_test)
#     mlflow.log_params(best_rf_params)

# # Logistic Regression
# with mlflow.start_run(run_name="logging_models_lr", experiment_id=experiment_id):
#     mlflow.sklearn.autolog()
#     lr_model = LogisticRegression(random_state=42, max_iter=200)
#     best_lr, best_lr_params = optimize_model(lr_model, lr_params, X_train_upsampled, Y_train_upsampled)
#     y_pred_lr = best_lr.predict(X_test)
#     mlflow.log_params(best_lr_params)

# # Gradient Boosting Classifier
# with mlflow.start_run(run_name="logging_models_gb", experiment_id=experiment_id):
#     mlflow.sklearn.autolog()
#     gb_model = GradientBoostingClassifier(random_state=42)
#     best_gb, best_gb_params = optimize_model(gb_model, gb_params, X_train_upsampled, Y_train_upsampled)
#     y_pred_gb = best_gb.predict(X_test)
#     mlflow.log_params(best_gb_params)


# create model version
source = "/home/grace/Projects_training_CDI/Mlops_project/BankingAnalysisMlflow/testing_mlflow_artifact/d678e9adf2214f24bbd9427897b3eaeb/artifacts/best_estimator"
run_id = "d678e9adf2214f24bbd9427897b3eaeb"

# create registered model
model_name="logging_models_rf_choosen"
# client.create_registered_model(model_name)

client.create_model_version(name=model_name, source=source, run_id=run_id)

# transition model version stage
client.transition_model_version_stage(name=model_name, version=1, stage="Production")
