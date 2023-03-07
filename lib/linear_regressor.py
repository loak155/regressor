import os
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from lib._base_regressor import BaseRegressor

logger = getLogger(__name__)


class LinearRegression_(BaseRegressor):

    linear_regression_params = {"n_jobs": None}

    def __init__(self, X=None, y=None, train_test_split_params=None, linear_regression_params=None):
        super().__init__(X, y, train_test_split_params)

        if linear_regression_params is not None:
            self.linear_regression_params = linear_regression_params

    def train(self, X_train=None, y_train=None):
        logger.info(f"Start: Train")
        if X_train is not None:
            self.X_train = X_train
        if y_train is not None:
            self.y_train = y_train

        model = LinearRegression(**self.linear_regression_params)
        model.fit(X_train, y_train)
        self.model = model
        logger.info(f"Finish: Train")

    def predict(self, X_test=None, y_test=None):
        logger.info(f"Start: Predict")
        if X_test is not None:
            self.X_test = X_test
        if y_test is not None:
            self.y_test = y_test

        self.pred = pd.Series(self.model.predict(self.X_test), index=self.X_test.index)
        self.actual_pred = pd.concat([self.y_test, self.pred], axis=1)
        self.actual_pred.columns = ["actual", "pred"]
        self.y_result = pd.concat([self.X_test, self.actual_pred], axis=1)
        r2 = self.model.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, self.pred)
        rmse = np.sqrt(mse)
        self.scores = {"r2": r2, "mse": mse, "rmse": rmse}
        logger.info(f"Score: {self.scores}")
        logger.info(f"Finish: Predict")

    def output(self, dir):
        logger.info(f"Start: Output")

        model_path = os.path.join(dir, "model.pkl")
        self.save_model(model_path)

        params_path = os.path.join(dir, "param.json")
        self.save_dict_to_json(self.model.get_params(), params_path)

        scores_path = os.path.join(dir, "scores.txt")
        self.save_dict_to_txt(self.scores, scores_path)

        pred_path = os.path.join(dir, "pred.csv")
        self.save_df_to_csv(self.y_result, pred_path)

        coef_and_intercept_path = os.path.join(dir, "coef_and_intercept.txt")
        coef_and_intercept = dict(zip(self.X_train.columns, self.model.coef_))
        coef_and_intercept["intercept"] = self.model.intercept_
        self.save_dict_to_txt(coef_and_intercept, coef_and_intercept_path)

        graph_path = os.path.join(dir, "graph_predicted_vs_actual.png")
        self.save_graph_predicted_vs_actual(graph_path)

        graph_path = os.path.join(dir, "graph_predicted_and_actual.png")
        self.save_graph_predicted_and_actual(graph_path)

        graph_path = os.path.join(dir, "graph_permutation_importance.png")
        self.save_graph_permutation_importance(graph_path)

        logger.info(f"Finish: Output")

