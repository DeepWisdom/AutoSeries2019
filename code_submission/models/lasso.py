from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from log_utils import log, timeclass
import numpy as np
import pandas as pd
from data_utils import downcast
from data_utils import get_rmse
import gc
from sklearn.preprocessing import Imputer


class LassoModel:
    def __init__(self):
        self.enc = {}
        self.cat2idx = {}
        self.dist = {}
        self.scaler = StandardScaler()
        self.model = None
        self.best_alpha = 1
        self.good_cols = None
        self.good_idxs = None
        self.train_shape = None
        self.do_sample = False
        self.imputer = Imputer()
        self.size =1000

    #@timeclass(cls='RidgeModel')
    def fit(self, X, y, categories):
        log(f'debug{self.good_cols}')
        if self.good_cols is not None:
            X = X[self.good_cols]
            self.cat_cols = tuple([col for col in categories if col in self.good_cols])
            self.num_cols = tuple([col for col in X.columns if col not in categories])
        else:
            self.cat_cols = tuple(categories)
            self.num_cols = tuple([col for col in X.columns if col not in categories])

        if self.train_shape is None:
            self.train_shape = X.shape[0]

        X_sample = X.iloc[:self.train_shape].sample(frac=0.8, random_state=2020)
        y_sample = y.loc[X_sample.index]

        X_test = X.iloc[self.train_shape:]
        y_test = y.loc[self.train_shape:]

        X = pd.concat([X_sample, X_test], axis=0)
        y = pd.concat([y_sample, y_test], axis=0)

        del X_sample, y_sample, X_test, y_test
        gc.collect()

        # self.cat_cols = tuple()
        # self.num_cols = tuple([col for col in X.columns])

        #log(f'train num col: {self.num_cols}')
        #log(f'train cat col:{self.cat_cols}')

        cat_feats = self.cat_fit_transform(X, mode='fit_trans')
        num_feats = self.num_fit_transform(X, mode='fit_trans')

        if len(cat_feats) > 0 and len(num_feats) > 0:
            feats = np.concatenate([cat_feats, num_feats], axis=1)
        elif len(cat_feats) > 0:
            feats = cat_feats
        elif len(num_feats) > 0:
            feats = num_feats
        #log(f'before downcast {feats.dtype}')
        feats = downcast(feats)

        self.model = Lasso(alpha=self.best_alpha, max_iter=500)

        try:
            if not self.do_sample:
                self.model.fit(feats, y)
            else:
                self.model.fit(feats[-self.size:], y.iloc[-self.size:])
        except:
            try:
                m = int(feats.shape[0]/2)
                self.model.fit(feats[-m:], y.iloc[-m:])
                self.do_sample = True
                self.size = m
            except:
                m = min(int(feats.shape[0]/5), 500000)
                self.model.fit(feats[-m:], y.iloc[-m:])
                self.size = m
                self.do_sample = True
        return self

    #@timeclass(cls='RidgeModel')
    def predict(self, X):
        if self.good_cols is not None:
            X = X[self.good_cols]
        cat_feats = self.cat_fit_transform(X, mode='trans')
        num_feats = self.num_fit_transform(X, mode='trans')

        if len(cat_feats) > 0 and len(num_feats) > 0:
            feats = np.concatenate([cat_feats, num_feats], axis=1)
        elif len(cat_feats) > 0:
            feats = cat_feats
        elif len(num_feats) > 0:
            feats = num_feats
        return self.model.predict(feats)

    def cat_fit_transform(self, df, mode):
        if len(self.cat_cols) == 0:
            return []

        df = df[list(self.cat_cols)]
        res = []
        start = 0
        for cat in self.cat_cols:
            if mode == 'fit_trans':
                self.enc[cat] = OneHotEncoder(handle_unknown='ignore')
                arr = self.enc[cat].fit_transform(df[cat].values.reshape(-1, 1)).toarray()
                end = start + arr.shape[1]
                self.cat2idx[cat] = [i for i in range(start, end)]
                start = end
            elif mode == 'trans':
                if df[cat].isnull().sum() > 0:
                    arr = self.enc[cat].transform(df[cat].fillna(self.enc[cat].n_values_[0]).values.reshape(-1, 1)).toarray()
                else:
                    arr = self.enc[cat].transform(df[cat].values.reshape(-1, 1)).toarray()
            res.append(arr)
        if mode == 'fit_trans':
            self.cat2idx['end'] = end
        cat_feats = []
        if len(res) > 0:
            cat_feats = np.concatenate(res, axis=1)
        return cat_feats

    def num_fit_transform(self, df, mode):
        if not self.num_cols:
            return []
        df = df[list(self.num_cols)]
        arr = df.values
        if mode == 'fit_trans':
            imputed_arr = self.imputer.fit_transform(arr)
            norm_arr = self.scaler.fit_transform(imputed_arr)
        elif mode == 'trans':
            imputed_arr = self.imputer.transform(arr)
            norm_arr = self.scaler.transform(imputed_arr)
        return norm_arr

    def explore_params(self, X, y, categories):
        self.cat_cols = tuple(categories)
        self.num_cols = [col for col in X.columns if col not in categories]

        log(f'train num col: {self.num_cols}')
        log(f'train cat col:{self.cat_cols}')

        cat_feats = self.cat_fit_transform(X, mode='fit_trans')
        num_feats = self.num_fit_transform(X, mode='fit_trans')

        if len(cat_feats) > 0 and len(num_feats) > 0:
            feats = np.concatenate([cat_feats, num_feats], axis=1)
        elif len(cat_feats) > 0:
            feats = cat_feats
        elif len(num_feats) > 0:
            feats = num_feats
        log(f'before downcast {feats.dtype}')
        feats = downcast(feats)
        log(f'aft downcast {feats.dtype}')

        feats = feats[-50000:]
        y = y.iloc[-50000:]

        log(f'train features shape : {feats.shape}')

        X_train, X_eval, y_train, y_eval = train_test_split(feats, y, test_size=0.2, shuffle=False, random_state=0)

        self.select_cols(X_train, X_eval, y_train, y_eval)
        final_rmse = self.select_alpha(X_train, X_eval, y_train, y_eval)

        return final_rmse

    def log_feat_importances(self):
        importances = pd.DataFrame({'features': [f'c_{i}' for i in range(len(self.model.coef_)-len(self.num_cols))] + list(self.num_cols),
                                    'importances': self.model.coef_})

        importances.sort_values('importances', ascending=False, inplace=True)

        log('feat importance:')
        log(f'{importances.head(100)}')

    def select_cols(self,  X_train, X_eval, y_train, y_eval):
        model = Lasso(alpha=1, max_iter=500)
        model.fit(X_train, y_train)
        preds = model.predict(X_eval)

        start_rmse = get_rmse(preds, y_eval)
        log(f'valid rmse before select_cols: {start_rmse}')
        bad_cols = []
        bad_idxs = []
        cat_bad_cols = []
        min_rmse = start_rmse
        tol = 0.05
        for col, idxs in self.cat2idx.items():
            if col == 'end':
                continue
            new_train = np.delete(X_train, idxs, axis=1)
            new_eval = np.delete(X_eval, idxs, axis=1)
            model = Lasso(alpha=1, max_iter=500)
            model.fit(new_train, y_train)
            preds = model.predict(new_eval)
            rmse = get_rmse(preds, y_eval)
            log(f'drop col {col} valid rmse: {rmse}')
            if rmse < (min_rmse-tol):
                cat_bad_cols.append(col)
                bad_cols.append(col)
                bad_idxs.extend(idxs)
        for idx in range(self.cat2idx['end'], X_train.shape[1]):
            col = self.num_cols[idx - self.cat2idx['end']]
            new_train = np.delete(X_train, idx, axis=1)
            new_eval = np.delete(X_eval, idx, axis=1)
            model = Lasso(alpha=1, max_iter=500)
            model.fit(new_train, y_train)
            preds = model.predict(new_eval)
            rmse = get_rmse(preds, y_eval)
            log(f'drop col {col} valid rmse: {rmse}')
            if rmse < (min_rmse-tol):
                bad_cols.append(col)
                bad_idxs.append(idx)

        good_cols = [i for i in (list(self.cat_cols) + list(self.num_cols)) if i not in bad_cols]
        good_idxs = [i for i in range(X_train.shape[1]) if i not in bad_idxs]
        model = Lasso(alpha=1, max_iter=500)
        model.fit(X_train[:, good_idxs], y_train)
        preds = model.predict(X_eval[:, good_idxs])

        self.sel_cols_rmse = get_rmse(preds, y_eval)
        log(f'valid rmse after select_cols: {self.sel_cols_rmse}')

        if self.sel_cols_rmse < start_rmse:
            self.good_cols = good_cols
            self.good_idxs = good_idxs
            self.cat_bad_cols = cat_bad_cols
        else:
            self.cat_bad_cols = []

        log(f'bad cols: {bad_cols}')

    def select_alpha(self, X_train, X_eval, y_train, y_eval):
        if self.good_cols is not None:
            X_train = X_train[:, self.good_idxs]
            X_eval = X_eval[:, self.good_idxs]

        alpha = 1
        best_rmse = self.sel_cols_rmse

        for i in np.arange(1, 4, 1):
            model = Lasso(alpha=i, max_iter=500)
            model.fit(X_train, y_train)
            preds = model.predict(X_eval)
            rmse = get_rmse(preds, y_eval)
            log(f'alpha {i} valid rmse: {rmse}')
            if rmse < best_rmse:
                alpha = i
                best_rmse = rmse

        for i in np.arange(max(0.1, alpha-0.6), alpha+0.6, 0.1):
            model = Lasso(alpha=i, max_iter=500)
            model.fit(X_train, y_train)
            preds = model.predict(X_eval)
            rmse = get_rmse(preds, y_eval)
            log(f'{i} valid rmse: {rmse}')
            if rmse < best_rmse:
                alpha = i
                best_rmse = rmse

        self.best_alpha = alpha
        log(f'after select_alpha rmse {best_rmse}')
        return best_rmse














