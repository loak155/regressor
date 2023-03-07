import os
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from lib._base_regressor import BaseRegressor

logger = getLogger(__name__)


class SVR_(BaseRegressor):
    grid_search_cv_params = {
        "n_jobs": -1,
        "cv": None,
        "verbose": 3,
    }
    svr_params = {
        "kernel": ["rbf"],
        "degree": [3],
    }

    def __init__(self, X=None, y=None, train_test_split_params=None, grid_search_cv_params=None, svr_params=None):
        super().__init__(X, y, train_test_split_params)

        if grid_search_cv_params is not None:
            self.grid_search_cv_params = grid_search_cv_params
        if svr_params is not None:
            self.svr_params = svr_params

    def grid_search(self, X_train=None, y_train=None):
        logger.info(f"Start: Grid Search")
        logger.info(f"Grid Search Params: {self.svr_params}")

        if X_train is not None:
            self.X_train = X_train
        if y_train is not None:
            self.y_train = y_train

        gscv = GridSearchCV(SVR(), self.svr_params, **self.grid_search_cv_params)
        gscv.fit(X_train, y_train)
        logger.info(f"Best Score: {gscv.best_score_}")
        for key, value in gscv.best_params_.items():
            logger.info(f"Best Param: {key} = {value}")
        self.params = gscv.best_params_
        logger.info(f"Finish: Grid Search")

    def train(self, X_train=None, y_train=None):
        logger.info(f"Start: Train")
        if X_train is not None:
            self.X_train = X_train
        if y_train is not None:
            self.y_train = y_train

        if not hasattr(self, "params"):
            self.grid_search(X_train, y_train)

        model = SVR(**self.params)
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

        graph_path = os.path.join(dir, "graph_predicted_vs_actual.png")
        self.save_graph_predicted_vs_actual(graph_path)

        graph_path = os.path.join(dir, "graph_predicted_and_actual.png")
        self.save_graph_predicted_and_actual(graph_path)

        graph_path = os.path.join(dir, "graph_permutation_importance.png")
        self.save_graph_permutation_importance(graph_path)

        logger.info(f"Finish: Output")
