# imports
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

from memoized_property import memoized_property

from mlflow.tracking import MlflowClient
import mlflow

import joblib


class Trainer():

    MLFLOW_URI = "https://mlflow.lewagon.co/"
    myname = "PierreB"
    EXPERIMENT_NAME = f"TaxifareModel_{myname}"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self, estimator):
        """defines the pipeline as a class attribute"""
        distance_pipeline = make_pipeline(DistanceTransformer(), StandardScaler())  # noqa: E501
        time_pipeline = make_pipeline(TimeFeaturesEncoder("pickup_datetime"), OneHotEncoder(handle_unknown="ignore"))  # noqa: E501

        dp_columns = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]  # noqa: E501
        tp_columns = ["pickup_datetime"]

        # create preprocessing pipeline
        preproc_pipeline = make_column_transformer((distance_pipeline, dp_columns),  # noqa: E501
                                                   (time_pipeline, tp_columns))

        self.pipeline = make_pipeline(preproc_pipeline, estimator)

    def run(self, estimator):
        """set and train the pipeline"""
        self.set_pipeline(estimator)
        self.pipeline = self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        return compute_rmse(self.pipeline.predict(X_test), y_test)

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'trainer_model.joblib')

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id  # noqa: E501

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    dataset_size = 20000
    taxi_df = get_data(nrows=dataset_size)

    # clean data
    taxi_df = clean_data(taxi_df)

    # set X and y
    X = taxi_df.drop(columns="fare_amount")
    y = taxi_df.fare_amount

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # noqa: E501

    # try different estimators
    estimators = {"LinReg": LinearRegression(),
                  "Lasso": Lasso(),
                  "Ridge": Ridge(),
                  "KNNreg": KNeighborsRegressor(),
                  "RandForest": RandomForestRegressor()}

    for estimator_name, estimator_object in estimators.items():
        # train
        my_trainer = Trainer(X_train, y_train)
        my_trainer.run(estimator_object)

        # evaluate
        rmse = my_trainer.evaluate(X_test, y_test)
        my_trainer.mlflow_log_metric("rmse", rmse)
        my_trainer.mlflow_log_param("model", estimator_name)
        my_trainer.mlflow_log_param("nrows", dataset_size)
        print("RMSE :", rmse)

        my_trainer.save_model()
