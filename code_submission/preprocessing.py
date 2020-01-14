import pandas as pd
import numpy as np


def parse_time(xtime: pd.Series):
    result = pd.DataFrame()

    dtcol = pd.to_datetime(xtime, unit='s')

    result[f'{xtime.name}'] = dtcol.astype('int64') // 10**9
    result[f'{xtime.name}_year'] = dtcol.dt.year
    result[f'{xtime.name}_month'] = dtcol.dt.month
    result[f'{xtime.name}_day'] = dtcol.dt.day
    result[f'{xtime.name}_weekday'] = dtcol.dt.weekday
    result[f'{xtime.name}_hour'] = dtcol.dt.hour

    return result


class TypeAdapter:
    def __init__(self, primitive_cat):
        self.adapt_cols = primitive_cat.copy()

    def fit_transform(self, X):
        cols_dtype = dict(zip(X.columns, X.dtypes))

        for key, dtype in cols_dtype.items():
            if dtype == np.dtype('object'):
                self.adapt_cols.append(key)
            if key in self.adapt_cols:
                X[key] = X[key].apply(hash_m)

        return X

    def transform(self, X):
        for key in X.columns:
            if key in self.adapt_cols:
                X[key] = X[key].apply(hash_m)

        return X


def hash_m(x):
    return hash(x) % 1048575
