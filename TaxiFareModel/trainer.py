# imports
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        distance_pipeline = make_pipeline(DistanceTransformer(), StandardScaler())  # noqa: E501
        time_pipeline = make_pipeline(TimeFeaturesEncoder("pickup_datetime"), OneHotEncoder(handle_unknown="ignore"))  # noqa: E501

        dp_columns = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]  # noqa: E501
        tp_columns = ["pickup_datetime"]

        # create preprocessing pipeline
        preproc_pipeline = make_column_transformer((distance_pipeline, dp_columns),  # noqa: E501
                                                   (time_pipeline, tp_columns))

        self.pipeline = make_pipeline(preproc_pipeline, LinearRegression())

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline = self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        return compute_rmse(self.pipeline.predict(X_test), y_test)


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
