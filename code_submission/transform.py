import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from log_utils import log, timeclass
from data_utils import downcast
import CONSTANT




class Transformer:
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X):
        pass



class CatTransformer(Transformer):
    def __init__(self):
        # 禁止打乱顺序，顺序与编码有关
        self.cats = []

    def fit(self, ss):
        cats = ss.dropna().drop_duplicates().values

        if len(self.cats) == 0:
            self.cats = sorted(list(cats))
        else:
            added_cats = sorted(set(cats) - set(self.cats))
            self.cats.extend(added_cats)

    def transform(self, ss):
        codes = pd.Categorical(ss, categories=self.cats).codes + CONSTANT.CAT_SHIFT
        # more = set(self.cats) - set(ss)
        # print(f'more:{more}')

        codes = codes.astype('float')
        codes[codes == (CONSTANT.CAT_SHIFT - 1)] = np.nan

        # nan_ratio = np.isnan(codes).mean()
        # print(f'nan_ratio')
        # print(nan_ratio)

        codes = downcast(codes, accuracy_loss=False)
        return codes

    def fit_transform(self, ss):
        self.fit(ss)
        return self.transform(ss)


class NumTransformer(Transformer):
    def fit(self, ss):
        pass

    def transform(self, ss):
        return downcast(ss)

    def fit_transform(self, ss):
        return self.transform(ss)
