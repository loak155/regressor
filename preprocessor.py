# 正規化、標準化

# https://di-acc2.com/programming/python/3748/#index_id5
# https://data-analysis-stats.jp/%E3%83%87%E3%83%BC%E3%82%BF%E5%89%8D%E5%87%A6%E7%90%86/%E6%AD%A3%E8%A6%8F%E5%8C%96%E3%81%A8%E6%A8%99%E6%BA%96%E5%8C%96%E3%81%AA%E3%81%A9%E3%81%AE%E7%89%B9%E5%BE%B4%E9%87%8F%E3%81%AE%E3%82%B9%E3%82%B1%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0%EF%BC%88feature-sca/
# https://aiacademy.jp/texts/show/?id=187
# https://qiita.com/yShig/items/dbeb98598abcc98e1a57

# 決定木の場合は、export_graphvizも出力する

from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler

logger = getLogger(__name__)


class Preprocessor(object):
    def __init__(self):
        pass


class NumericPreprocessor(object):
    def __init__(self):
        pass

    def normalization(self, data, min=None, max=None):
        """正規化：特徴量の値の範囲を一定の範囲(0~1が多い)に収める
            ・最大値及び最小値が決まっている場合。(例：テストの点数など)
            args:
                data (DataFrame or Series)
                min (list or int)
                max (list or int)
        """
        scaler = MinMaxScaler()
        if type(data) is pd.core.frame.DataFrame:
            if min is not None and max is not None:
                min_max = np.array([min, max])
                scaler.fit(min_max)
                transformed_data = scaler.transform(data)
            else:
                transformed_data = scaler.fit_transform(data)
        elif type(data) is pd.core.series.Series:
            if min is not None and max is not None:
                min_max = np.array([min, max]).reshape([-1, 1])
                scaler.fit(min_max)
                transformed_data = scaler.transform(data.values.reshape([-1, 1]))
            else:
                transformed_data = scaler.fit_transform(data.values.reshape([-1, 1]))
        return transformed_data, scaler

    def standardization(self, df):
        """標準化：特徴量の平均を0, 分散を1にする。こちらを使うことが多い。
            ・最大値及び最小値が決まっていない場合
            ・外れ値が存在する場合
        """
        scaler = StandardScaler()
        transformed_data = scaler.fit_transform(df)
        return transformed_data, scaler

    def impute(self, df, threshold=0.5, strategy="mean"):
        """欠損値処理"""
        missing_count = df.isnull().sum()
        self.logger.info(f"欠損値カウント:{missing_count.to_dict()}")

        index_length = len(df)
        missing_percentage = missing_count / index_length
        self.logger.info(f"欠損値割合:{missing_percentage.to_dict()}")

        missing_percentage = missing_percentage[missing_percentage > 0]
        columns_above_threshold = missing_percentage[missing_percentage >= threshold].index
        columns_below_threshold = missing_percentage[missing_percentage < threshold].index

        # 閾値以上の場合、列を削除
        if len(columns_above_threshold) > 0:
            self.logger.info(f"欠損値割合が閾値以上のため、削除する")
            self.logger.info(columns_above_threshold.values)
            df.drop(columns=columns_above_threshold, inplace=True)
        # 閾値以下の場合、欠損値埋め
        imputer = None
        if len(columns_below_threshold) > 0:
            self.logger.info(f"欠損値割合が閾値以下のため、欠損地埋めする")
            self.logger.info(columns_below_threshold.values)
            imputer = SimpleImputer(strategy=strategy)
            transformed_data = imputer.fit_transform(df.loc[:, columns_below_threshold])
            df.loc[:, columns_below_threshold] = transformed_data
        return df, imputer

    def outlier_handling(self, df):
        """外れ値処理(欠損値処理の後にする)"""
        clf = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
        clf.fit(df)
        df["predict"] = clf.predict(df)
        df_without_outliers = df[df["predict"] == 1]
        df_without_outliers.drop(columns="predict", inplace=True)
        return df_without_outliers


class CategoricalPreprocessor(object):
    """カテゴリ変数系特徴量の前処理"""

    def __init__(self, df):
        self.df_category = df.select_dtypes(include="object")

    def category_encoders(self, df):
        """カテゴリ変数系特徴量の前処理"""
        # https://zenn.dev/megane_otoko/articles/2021ad_06_category_encoding

        import category_encoders as ce
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

        encoder = OneHotEncoder()
        encoder.fit_transform(self.df_category)
        self.df_category = encoder.transform(self.df_category)

        # ce.OrdinalEncoer()  # LabelEncoderと同じ。欠損値の処理ができる。
        # OneHotEncoder()  # 種類が少ない場合、基本はOneHotEncoder
        # LabelEncoder()  # SMLのように < などで表せる場合
        # BinaryEncoder()  # 種類が多い場合
        # https://www.salesanalytics.co.jp/datascience/datascience039/


def main():
    df = pd.DataFrame([["black", "yes", 1], ["white", "no", 1], ["yellow", "no", 1], ["black", "yes", 1]], columns=["label", "binary", "idx"])
    print(df)
    preprocessor = CategoricalPreprocessor(df)
    preprocessor.category_encoders(df)
    pass


if __name__ == "__main__":
    main()
