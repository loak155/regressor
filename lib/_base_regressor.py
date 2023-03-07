import datetime
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.25


class BaseRegressor(object):
    train_test_split_params = {"test_size": None, "random_state": None, "shuffle": True}

    def __init__(self, X=None, y=None, train_test_split_params=None):
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y
        if train_test_split_params is not None:
            self.train_test_split_params = train_test_split_params

    def make_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, **self.train_test_split_params)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        with open(path, mode="rb") as f:
            self.model = pickle.load(f)

    def save_dict_to_json(self, dict, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict, f, indent=2)

    def load_json(self, path):
        with open(path, "r") as f:
            dict = json.load(f)
        return dict

    def save_dict_to_txt(self, dict, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for key, value in dict.items():
                f.write(f"{key}: {value}\n")

    def save_df_to_csv(self, df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)

    def template_path(self, path, item):
        if not hasattr(self, "timestamp"):
            self.timestamp = datetime.datetime.now()

        template_value = {
            "algorithm": item["algorithm"],
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

    def save_graph_predicted_vs_actual(self, path):
        plt.figure(figsize=(19.2, 10.8), tight_layout=True)
        plt.title("Predicted vs. Actual")
        plt.scatter(self.pred, self.y_test)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.grid(True)
        # 最小二乗近似
        coef = np.polyfit(self.pred.sort_index(), self.y_test.sort_index(), 1)
        y_fit = coef[0] * self.pred.sort_index() + coef[1]
        y_fit.index = self.pred.sort_index()
        y_fit.sort_index(inplace=True)
        plt.plot(y_fit, label="least square", color="g")
        plt.legend(loc="lower right")
        plt.savefig(path)
        plt.clf()
        plt.close()

    def save_graph_predicted_and_actual(self, path):
        plt.figure(figsize=(19.2, 10.8), tight_layout=True)
        plt.title("Predicted & Actual")
        plt.plot(self.pred.sort_index(), label="Predicted")
        plt.plot(self.y_test.sort_index(), label="Actual")
        plt.legend(loc="upper right")
        plt.xlabel("Index")
        plt.ylabel("Predicted & Actual")
        plt.grid(True)
        plt.savefig(path)
        plt.clf()
        plt.close()

    def save_graph_permutation_importance(self, path):
        plt.figure(figsize=(19.2, 10.8), tight_layout=True)
        plt.title("Permutation Importance (test set)")
        result = permutation_importance(self.model, self.X_test, self.y_test)
        sorted_index = result.importances_mean.argsort()
        plt.boxplot(result.importances[sorted_index].T, vert=False, labels=np.array(self.X_test.columns)[sorted_index])
        plt.savefig(path)
        plt.clf()
        plt.close()
