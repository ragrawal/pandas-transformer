Pandas-Transformer
==================

This module provides wrapper around sklearn-pandas library to help achieve:
1. Sequence of transformation operations on original columns as well as
derived columns.
2. Enable injecting custom functions (such as np.log, etc) without
constructing explicit classes that inherit TransformerMixin

Usage
------

>>> import numpy as np
>>> import pandas as pd
>>> from sklearn import datasets
>>> from pandas_transformer import PandasTransformer
>>> boston = datasets.load_boston()
>>> data = pd.DataFrame(boston.data, columns=boston.feature_names)
>>> transformer = PandasTransformer()
>>> transformer.apply('AGE', np.log1p, {'alias': 'age_log'})
>>> transformer.apply('age_log', np.exp, {'alias': 'age_exp'})
>>> transformer.apply('RM', [np.log1p, np.exp], {'alias': 'derived_rm'})
>>> derivedDF = transformer.fit_transform(data)




