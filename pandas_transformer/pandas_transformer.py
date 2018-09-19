import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas import DataFrameMapper

class PandasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__blocks = []

    def apply(self, columns, methods, options):
        """
        Add new transformation block
        :param columns: column names
        :type: str or list
        :param methods: list of transformers
        :type: function or list of functions
        :param options: parameters for sklearn-pandas data frame mapper
        :type: dict
        :return:
        """
        mapper = DataFrameMapper(
            [(columns, methods, options)]
            , df_out=True
            , default=False
        )

        self.__blocks.append(mapper)

        return self

    @property
    def blocks(self):
        return self.__blocks

    def __transform(self, x, y=None, fit=True):

        features = None
        for block in self.blocks:
            if isinstance(block.features[0][0], list):
                required_columns = set(block.features[0][0])
            else:
                required_columns = set([block.features[0][0]])

            original_columns = set(x.columns) & required_columns
            feature_columns = required_columns - original_columns

            if len(feature_columns) == 0:
                data = x[list(required_columns)]
            else:
                data = pd.concat(
                    [x[list(original_columns)], features[list(feature_columns)]]
                    , ignore_index=False
                    , axis=1
                )

            if fit is True:
                df = block.fit_transform(data)
            else:
                df = block.transform(data)

            df.index = data.index

            if features is not None:
                features = pd.concat([features, df], ignore_index=False, axis=1)
            else:
                features = df.copy()

        return features

    def transform(self, x, y=None):
        return self.__transform(x, y, fit=False)

    def fit(self, x, y=None):
        self.__transform(x, y, fit=True)
        return self