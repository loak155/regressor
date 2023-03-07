# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

# 主成分分析(PCA), 主成分回帰(PCR), 部分的最小二乗法(PLS)
# https://qiita.com/oki_kosuke/items/43cb63134f9a03ebc79a
# https://qiita.com/maskot1977/items/082557fcda78c4cdb41f
# https://www.takapy.work/entry/2019/02/08/002738
# https://dcross.impress.co.jp/docs/column/column20170926-02/001710.html
# https://zenn.dev/wsuzume/articles/0c51b06e774c2e
# https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/%E7%B5%B1%E8%A8%88%E5%AD%A6/%E4%B8%BB%E6%88%90%E5%88%86%E5%9B%9E%E5%B8%B0%EF%BC%88pcr%EF%BC%89%E3%81%AE%E3%83%A1%E3%83%A2/
# https://www.salesanalytics.co.jp/datascience/datascience118
# https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/%E7%B5%B1%E8%A8%88%E5%AD%A6/%E9%83%A8%E5%88%86%E7%9A%84%E6%9C%80%E5%B0%8F%EF%BC%92%E4%B9%97%E6%B3%95%EF%BC%88pls%EF%BC%89%E3%81%AE%E3%83%A1%E3%83%A2/


# 次元削減
# 次元削除の手法のひとつにPCA、特徴選択がある。

# 特徴量選択
# https://qiita.com/shimopino/items/5fee7504c7acf044a521
# https://qiita.com/FukuharaYohei/items/db88a8f4c4310afb5a0d
# https://rightcode.co.jp/blog/informatin-technology/feature-selection-right-choice

# sklearn.feature_selection
# https://qiita.com/rockhopper/items/a68ceb3248f2b3a41c89
# https://qiita.com/nanairoGlasses/items/d7d4c190d11ba663635d

# https://runebook.dev/ja/docs/scikit_learn/modules/feature_selection
# https://zenn.dev/megane_otoko/articles/2021ad_18_select_features


from logging import getLogger

logger = getLogger(__name__)


class DimensionalityReduction(object):
    def __init__(self):
        pass

    def variance_threshold(self, df, threshold=0):
        """"分散が閾値以下の項目を削除"""
        from sklearn.feature_selection import VarianceThreshold

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df)
        delete_columns = [df.columns[i] for i, bool in enumerate(selector.get_support()) if not bool]
        logger.info(f"分散が閾値以下の項目を削除します。")
        logger.info(f"閾値: {threshold}, 削除項目: {delete_columns}")  # 分散も表示する
        return df.loc[:, selector.get_support()]

    def correlation_coefficient_threshold(self, df, threshold=0.9):
        """"相関係数が閾値以上の項目を削除"""
        feat_corr = set()
        corr_matrix = df.corr()
        for i, column in enumerate(corr_matrix.columns):
            for j, index in enumerate(corr_matrix.index):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    feat_name = corr_matrix.columns[i]
                    feat_corr.add(feat_name)
        print(feat_corr)
