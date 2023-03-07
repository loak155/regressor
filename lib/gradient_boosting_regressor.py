import os
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit

from lib._base_regressor import BaseRegressor

logger = getLogger(__name__)


class GradientBoostingRegressor_(BaseRegressor):

    grid_search_cv_params = {
        "n_jobs": -1,
        "cv": None,
        "verbose": 3,
    }
    gradient_boosting_regressor_params = {
        "loss": ["ls"],
        "learning_rate": [0.1],
        "n_estimators": [100],
        "subsample": [1.0],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_depth": [3],
        "max_features": [None],
    }

    def __init__(self, X=None, y=None, train_test_split_params=None, grid_search_cv_params=None, gradient_boosting_regressor_params=None):
        super().__init__(X, y, train_test_split_params)

        if grid_search_cv_params is not None:
            self.grid_search_cv_params = grid_search_cv_params
        if gradient_boosting_regressor_params is not None:
            self.gradient_boosting_regressor_params = gradient_boosting_regressor_params

    def grid_search(self, X_train=None, y_train=None):
        logger.info(f"Start: Grid Search")
        logger.info(f"Grid Search Params: {self.gradient_boosting_regressor_params}")

        if X_train is not None:
            self.X_train = X_train
        if y_train is not None:
            self.y_train = y_train

        gscv = GridSearchCV(GradientBoostingRegressor(), self.gradient_boosting_regressor_params, **self.grid_search_cv_params)
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

        model = GradientBoostingRegressor(**self.params)
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

        graph_path = os.path.join(dir, "graph_feature_importance.png")
        self.save_graph_feature_importance(graph_path)

        graph_path = os.path.join(dir, "graph_permutation_importance.png")
        self.save_graph_permutation_importance(graph_path)

        logger.info(f"Finish: Output")

    def save_graph_feature_importance(self, path):
        plt.figure(figsize=(19.2, 10.8), tight_layout=True)
        plt.title("Feature Importance (MDI)")
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0])
        plt.barh(pos, feature_importance[sorted_idx], align="center")
        plt.yticks(pos, np.array(self.X_test.columns)[sorted_idx])
        plt.savefig(path)
        plt.clf()
        plt.close()

    # def cross_valid(self, X, y):
    #     model_list = []
    #     pred_list = []
    #     r2_list = []
    #     mse_list = []
    #     rmse_list = []

    #     self.grid_search(X, y.values.ravel())

    #     kf = KFold(n_splits=self.n_splits)
    #     cv_no = 1
    #     for train_index, test_index in kf.split(X, y):
    #         logger.info(f"Cross Validation {cv_no}/{self.n_splits}")
    #         X_train = X.iloc[train_index]
    #         X_test = X.iloc[test_index]
    #         y_train = y.iloc[train_index].values.ravel()
    #         y_test = y.iloc[test_index].values.ravel()

    #         self.train(X_train, y_train)
    #         pred, r2, mse, rmse = self.predict(X_test, y_test)

    #         model_list.append(self.model)
    #         pred_list.append(pred)
    #         r2_list.append(r2)
    #         mse_list.append(mse)
    #         rmse_list.append(rmse)

    #         cv_no += 1

    #     r2_avg = np.mean(r2_list)
    #     mse_avg = np.mean(mse_list)
    #     rmse_avg = np.mean(rmse_list)
    #     logger.info(f"average score: r2: {r2_avg}, mse: {mse_avg}, rmse: {rmse_avg}")

    #     return model_list, pred_list, r2_list, mse_list, rmse_list, r2_avg, mse_avg, rmse_avg

    # def cross_valid_of_time_series(self, X, y):
    #     model_list = []
    #     pred_list = []
    #     r2_list = []
    #     mse_list = []
    #     rmse_list = []

    #     self.grid_search(X, y)

    #     tscv = TimeSeriesSplit(n_splits=self.n_splits)
    #     cv_no = 1
    #     for train_index, test_index in tscv.split(X, y):
    #         logger.info(f"Cross Validation {cv_no}/{self.n_splits}")
    #         X_train = X.iloc[train_index]
    #         X_test = X.iloc[test_index]
    #         y_train = y.iloc[train_index]
    #         y_test = y.iloc[test_index]

    #         self.train(X_train, y_train)
    #         pred, r2, mse, rmse = self.predict(X_test, y_test)
    #         logger.info(f"r2: {r2}, mse: {mse}, rmse: {rmse}")

    #         model_list.append(self.model)
    #         pred_list.append(pred)
    #         r2_list.append(r2)
    #         mse_list.append(mse)
    #         rmse_list.append(rmse)

    #         cv_no += 1

    #     r2_avg = np.mean(r2_list)
    #     mse_avg = np.mean(mse_list)
    #     rmse_avg = np.mean(rmse_list)
    #     logger.info(f"average score: r2: {r2_avg}, mse: {mse_avg}, rmse: {rmse_avg}")

    #     return model_list, pred_list, r2_list, mse_list, rmse_list, r2_avg, mse_avg, rmse_avg

