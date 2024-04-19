from mlflow_utils import create_mlflow_experiment
from mlflow import MlflowClient
import mlflow
from mlflow.types.schema import Schema
from mlflow.types.schema import ColSpec
from sklearn.ensemble import RandomForestRegressor

if __name__=="__main__":

    experiment_id = create_mlflow_experiment(
        experiment_name="model_registry_grace",
        artifact_location="model_registry_artifacts",
        tags={"purpose": "learning"},
    )


    print(experiment_id)

    client = MlflowClient()
    model_name = "Test_grace"


    #with mlflow.start_run(run_name="first_run_registry_grace") as run:
        #mlflow.sklearn.log_model(artifact_path="rfr_model", sk_model=RandomForestRegressor(), registered_model_name="RandomForestRegressor")
        # create registered model
        #client.create_registered_model(model_name)

    # # create model version
    source = "file:/home/grace/Projects_training_CDI/Training/MLOPS_Course/model_registry_artifacts/5c07419ffebc4300b6f920f7861b1d88/artifacts/rfr_model"
    run_id = "5c07419ffebc4300b6f920f7861b1d88"
    #client.create_model_version(name=model_name, source=source, run_id=run_id)

    # # transition model version stage
    client.transition_model_version_stage(name=model_name, version=1, stage="Staging")
    client.transition_model_version_stage(name=model_name, version=2, stage="Production")

    # # delete model version
    # client.delete_model_version(name=model_name, version=1)

    # # delete registered model
    # client.delete_registered_model(name=model_name)

    # adding description to registired model.
    #client.update_registered_model(name=model_name, description="This is a test model")

    # adding tags to registired model.
    client.set_registered_model_tag(name=model_name, key="tag1", value="value1")

    # adding description to model version.
    client.update_model_version(name=model_name, version=1, description="This is a test model version")

    # adding tags to model version.
    client.set_model_version_tag(name=model_name, version=1, key="tag1", value="value1")







import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__=="__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="logging_models", experiment_id=experiment.experiment_id) as run:


        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

        # log model using autolog
        mlflow.autolog()
        # mlflow.sklearn.autolog()

        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)


        # log model
        # mlflow.sklearn.log_model(sk_model=rfc, artifact_path="random_forest_classifier")


        # print info about the run
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))


