import datetime
import os
from logging import config, getLogger

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib.gradient_boosting_regressor import GradientBoostingRegressor_
from lib.linear_regressor import LinearRegression_
from lib.mlp_regressor import MLPRegressor_
from lib.ramdom_forest_regressor import RandomForestRegressor_
from lib.support_vector_regressor import SVR_
from lib.xgboost_regressor import XGBRegressor_
from preprocessor import NumericPreprocessor

logger = getLogger(__name__)


class Regressor(object):
    def __init__(self, conf):
        self.conf = conf

        self.regressors = {}
        if "GradientBoostingRegressor" in conf["algorithm"]:
            self.regressors["GradientBoostingRegressor"] = GradientBoostingRegressor_(
                train_test_split_params=conf.get("train_test_split_params", None),
                grid_search_cv_params=conf.get("grid_search_cv_params", None),
                gradient_boosting_regressor_params=conf.get("gradient_boosting_regressor_params", None),
            )
        if "LinearRegression" in conf["algorithm"]:
            self.regressors["LinearRegression"] = LinearRegression_(
                train_test_split_params=conf.get("train_test_split_params", None), linear_regression_params=conf.get("linear_regression_params", None)
            )
        if "MLPRegressor" in conf["algorithm"]:
            self.regressors["MLPRegressor"] = MLPRegressor_(
                train_test_split_params=conf.get("train_test_split_params", None),
                grid_search_cv_params=conf.get("grid_search_cv_params", None),
                mlp_regressor_params=conf.get("mlp_regressor_params", None),
            )
        if "RandomForestRegressor" in conf["algorithm"]:
            self.regressors["RandomForestRegressor"] = RandomForestRegressor_(
                train_test_split_params=conf.get("train_test_split_params", None),
                grid_search_cv_params=conf.get("grid_search_cv_params", None),
                random_forest_regressor_params=conf.get("random_forest_regressor_params", None),
            )
        if "SVR" in conf["algorithm"]:
            self.regressors["SVR"] = SVR_(
                train_test_split_params=conf.get("train_test_split_params", None),
                grid_search_cv_params=conf.get("grid_search_cv_params", None),
                svr_params=conf.get("svr_params", None),
            )
        if "XGBRegressor" in conf["algorithm"]:
            self.regressors["XGBRegressor"] = XGBRegressor_(
                train_test_split_params=conf.get("train_test_split_params", None),
                grid_search_cv_params=conf.get("grid_search_cv_params", None),
                xgregressor_params=conf.get("xgregressor_params", None),
            )

    def load_csv(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def make_dataset(self, objective_variable_column):
        self.X = self.df.drop(self.df.columns[int(objective_variable_column)], axis=1)
        self.y = self.df.iloc[:, int(objective_variable_column)]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, **self.conf.get("train_test_split_params", None))

    def train_validation(self, output):
        for key, regressor in self.regressors.items():
            logger.info(f"regressor: {key}")

            regressor.train(self.X_train, self.y_train)
            regressor.predict(self.X_test, self.y_test)

            output_dir = self.template_path(output["dir"], key)
            regressor.output(output_dir)

    def template_path(self, path, algorithm):
        if not hasattr(self, "timestamp"):
            self.timestamp = datetime.datetime.now()

        template_value = {
            "algorithm": algorithm,
            "datetime": self.timestamp.strftime("%Y%m%d%H%M%S"),
            "date": self.timestamp.strftime("%Y%m%d"),
            "year": self.timestamp.strftime("%Y"),
            "month": self.timestamp.strftime("%m"),
            "day": self.timestamp.strftime("%d"),
            "h": self.timestamp.strftime("%H"),
            "m": self.timestamp.strftime("%M"),
            "s": self.timestamp.strftime("%S"),
        }

        return path.format(**template_value)

    @staticmethod
    def load_yaml(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data

    @staticmethod
    def get_logger_from_dict(conf):
        config.dictConfig(conf)
        logger = getLogger(__name__)
        return logger


def main():
    conf = Regressor.load_yaml("config.yaml")
    logger = Regressor.get_logger_from_dict(conf["logger"])
    regressor = Regressor(conf["regressor"])

    regressor.load_csv(conf["input"]["csv_path"])
    regressor.make_dataset(conf["input"]["objective_variable_column"])

    # regressor.train_validation(conf["output"])


if __name__ == "__main__":
    main()
