import gc
import numpy as np
from log_utils import log, timeclass
import pandas as pd
import CONSTANT
from feature.feat_opt import c_cats_combine


class Table:
    def __init__(self, X, info):
        self.info = info
        self.schema = info['schema']
        self.is_multivar = info['is_multivariate']

        self.label = info['label']
        self.id_cols = info['primary_id']
        self.cat_cols = [col for col, types in self.schema.items() if types == 'str']
        self.num_cols = [col for col, types in self.schema.items() if types == 'num' and col != self.label]
        self.init_num_cols = self.num_cols.copy()
        self.init_num_cols = []

        self.key_time_col = info['primary_timestamp']
        self.time_attr_cols = None

        self.num_multi_idx = len(self.id_cols)

        self.key_col = None
        self.key_set = None
        self.key2idx = None

        self.cat2idx = {}
        self.cat2info = {}
        self.encode_cat = []

        self.train_X = X
        self.train_y = X[self.label]
        self.test_X = None
        self.pred_X = None

        self.key_window = None
        self.cat2window = {}

        self.col2block = {}
        self.col2type = {}

        self.post_drop_set = set()

        self.col2max = {}

        self.cross_cols = set()

        self.init_key_info()
        self.init_col2type()

        #不能改变顺序
        self._save_cols = self.init_num_cols + [self.label]
        self.col2idx = {self._save_cols[i]: i for i in range(len(self._save_cols))}

        # log(f'num_cols:{self.num_cols}')
        # log(f'cat_cols:{self.cat_cols}')

    def fit_transform_output(self):
        X = self.train_X.copy()
        y = self.train_y.copy()
        self.drop_time_col(X)
        self.drop_label_col(X)
        #self.drop_key_cols(X)
        self.drop_post_drop_column(X)

        categories = self.get_categories(X)
        return X, y, categories

    def transform_output(self):
        self.drop_time_col(self.pred_X)
        #self.drop_key_cols(self.pred_X)
        self.drop_post_drop_column(self.pred_X)

        return self.pred_X

    def init_key_window(self):
        if self.key_num < CONSTANT.KEY_NUM:
            self.key_window = np.ones((self.key_num, len(self.init_num_cols) + 1, CONSTANT.MAX_LAG + 1)) * CONSTANT.MISSING
            if self.is_multivar:
                key_list = list(self.key_set)
                self.key2idx = {key_list[i]: i for i in range(self.key_num)}
                for key in self.key_set:
                    time_arr = self.train_X[self.train_X[self.key_col] == key].iloc[-CONSTANT.MAX_LAG-1:][self._save_cols].values
                    size = len(time_arr)
                    time_arr = np.swapaxes(time_arr, 0, 1)
                    idx = self.key2idx[key]
                    self.key_window[idx, :, -size:] = time_arr
            else:
                #print(f'init save cols {self._save_cols}')
                time_arr = self.train_X[self._save_cols].iloc[-CONSTANT.MAX_LAG - 1:].values
                size = len(time_arr)
                time_arr = np.swapaxes(time_arr, 0, 1)
                self.key_window[0, :, -size:] = time_arr
            #log(f'init key_window, shape is {self.key_window.shape}')
        else:
            if self.is_multivar:
                key_list = list(self.key_set)
                self.key2idx = {key_list[i]: i for i in range(self.key_num)}
                df = pd.pivot_table(self.train_X, index=[self.key_time_col], values=[self.label], columns=self.key_col,
                                    aggfunc=[np.mean])
                df = df.fillna(method="ffill")
                df = df.fillna(-1)

                persist_length = CONSTANT.MAX_LAG + 1
                self.key_window = np.ones(
                    (self.key_num, len(self.init_num_cols) + 1, CONSTANT.MAX_LAG + 1)) * CONSTANT.MISSING

                idx = np.array([self.key2idx[v] for v in df.columns.get_level_values(2)])
                time_arr = np.swapaxes(df.values[-persist_length:, :], 0, 1)
                size = time_arr.shape[1]
                self.key_window[idx, 0, -size:] = time_arr

                # import pdb;pdb.set_trace()
            else:
                # Todo: 还没改
                self.key_window = np.ones((self.key_num, len(self.init_num_cols) + 1, CONSTANT.MAX_LAG + 1)) * CONSTANT.MISSING
                #print(f'init save cols {self._save_cols}')
                time_arr = self.train_X[self._save_cols].iloc[-CONSTANT.MAX_LAG - 1:].values
                size = len(time_arr)
                time_arr = np.swapaxes(time_arr, 0, 1)
                self.key_window[0, :, -size:] = time_arr



    def add_key_window(self, new_hist):
        if self.key_num < CONSTANT.KEY_NUM:
            if not new_hist.empty:
                if self.num_multi_idx == 0:
                    time_arr = new_hist[self._save_cols].values
                    self.key_window[0, :, :-1] = self.key_window[:, :, 1:]
                    self.key_window[0, :, -1] = time_arr
                elif self.num_multi_idx >= 1:
                    if self.num_multi_idx == 1:
                        keys = new_hist[self.key_col].values
                    elif self.num_multi_idx > 1:
                        keys = c_cats_combine(new_hist[self.id_cols], self.col2max)
                    idxs = self.keys2idxs(keys)
                    time_arr = new_hist[self._save_cols].values
                    self.key_window[idxs, :, :-1] = self.key_window[idxs, :, 1:]
                    self.key_window[idxs, :, -1] = time_arr
        else:
            if not new_hist.empty:
                if self.num_multi_idx == 0:
                    time_arr = new_hist[self._save_cols].values
                    self.key_window[0, :, :-1] = self.key_window[:, :, 1:]
                    self.key_window[0, :, -1] = time_arr
                elif self.num_multi_idx >= 1:
                    if self.num_multi_idx == 1:
                        keys = new_hist[self.key_col].values
                    elif self.num_multi_idx > 1:
                        keys = c_cats_combine(new_hist[self.id_cols], self.col2max)
                    idxs = self.keys2idxs(keys)
                    time_arr = new_hist[self._save_cols].values
                    self.key_window[:, :, :-1] = self.key_window[:, :, 1:]
                    self.key_window[idxs, :, -1] = time_arr

    def keys2idxs(self, keys):
        return np.array([self.get_key2idx(key) for key in keys])

    def get_key2idx(self, key):
        if key in self.key2idx:
            return self.key2idx[key]
        else:
            #log(f'find new key: {key}')
            self.key_set.add(key)
            self.key2idx[key] = self.key_num
            self.key_num += 1

            init_new_arr = np.ones((1, len(self.init_num_cols) + 1, CONSTANT.MAX_LAG + 1)) * CONSTANT.MISSING
            self.key_window = np.append(self.key_window, init_new_arr, axis=0)

            return self.key2idx[key]

    def init_cat_window(self):
        df = self.train_X
        for cat in self.cat_cols:
            ss = df[cat]
            cat_num = ss.nunique()
            cat_set = set(ss.unique())
            info = {'num': cat_num, 'set': cat_set}
            self.cat2info[cat] = info
            cat_window = np.ones((cat_num, len(self.init_num_cols) + 1, CONSTANT.MAX_LAG + 1)) * CONSTANT.MISSING
            cat_list = list(cat_set)
            self.cat2idx[cat] = {cat_list[i]: i for i in range(cat_num)}
            group_df = df.groupby([cat, self.key_time_col])[self._save_cols].mean()
            for obj in cat_set:
                time_arr = group_df.loc[obj].iloc[-CONSTANT.MAX_LAG - 1:].values
                size = len(time_arr)
                time_arr = np.swapaxes(time_arr, 0, 1)
                idx = self.cat2idx[cat][obj]
                cat_window[idx, :, -size:] = time_arr
            self.cat2window[cat] = cat_window

    def add_cat_window(self, new_hist):
        if not new_hist.empty:
            for cat in self.cat_cols:
                mean_df = new_hist.groupby([cat])[self._save_cols].mean()
                objs = list(mean_df.index)
                idxs = self.objs2idxs(objs, cat)
                time_arr = mean_df.values
                hist = self.cat2window[cat]
                hist[idxs, :, :-1] = hist[idxs, :, 1:]
                hist[idxs, :, -1] = time_arr
                self.cat2window[cat] = hist

    def objs2idxs(self, objs, cat):
        return np.array([self.obj2idx(obj, cat) for obj in objs])

    def obj2idx(self, obj, cat):
        map_dict = self.cat2idx[cat]

        if obj in map_dict:
            return map_dict[obj]
        else:
            #log(f'find new obj in {cat}: {obj}')
            info = self.cat2info[cat]
            info['set'].add(obj)
            map_dict[obj] = info['num']
            info['num'] += 1
            self.cat2info = info
            self.cat2idx[cat] = map_dict

            init_new_arr = np.ones((1, len(self.init_num_cols) + 1, CONSTANT.MAX_LAG + 1)) * CONSTANT.MISSING
            cat_window = self.cat2window[cat]
            self.cat2window[cat] = np.append(cat_window, init_new_arr, axis=0)

            return map_dict[obj]

    def drop_time_col(self, df):
        if self.key_time_col is not None:
            #log(f'drop time col:{self.key_time_col}')
            df.drop(self.key_time_col, axis=1, inplace=True)
            gc.collect()

    def drop_id_col(self, df):
        if len(self.id_cols) != 0:
            #log(f'drop id col:{self.id_cols}')
            df.drop(self.id_cols, axis=1, inplace=True)
            gc.collect()

    def drop_label_col(self, df):
        if self.label in df.columns:
            df.drop(self.label, axis=1, inplace=True)
            gc.collect()

    def drop_key_cols(self, df):
        if len(self.id_cols) > 0:
            for col in self.id_cols:
                #log(f'drop id col:{self.id_cols}')
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
                    gc.collect()
        if self.key_col in df.columns:
            #log(f'drop key col:{self.key_col}')
            df.drop(self.key_col, axis=1, inplace=True)
            gc.collect()


    def drop_post_drop_column(self, df):
        if len(self.post_drop_set) != 0:
            drop_cols = list(self.post_drop_set)
            df.drop(drop_cols, axis=1, inplace=True)
            gc.collect()
            #log(f'post drop cols:{drop_cols}')

    def get_categories(self, df):
        categories = []
        col_set = set(df.columns)
        for col in self.cat_cols:
            if col in col_set:
                if df[col].nunique() <= 31:
                    categories.append(col)
        #log(f'get categories: {categories}')
        return categories

    def update_data(self, df, col2type, mode):
        if mode == 'train':
            self.train_X = df
            self.update_col2type(col2type)
        elif mode == 'pred':
            self.pred_X = df

    def update_col2type(self, col2type):
        self.col2type.update(col2type)
        self.type_reset()

    def type_reset(self):

        cat_cols = []
        num_cols = []

        for cname, ctype in self.col2type.items():
            if ctype == CONSTANT.CATEGORY_TYPE:
                cat_cols.append(cname)
            elif ctype == CONSTANT.NUMERICAL_TYPE:
                num_cols.append(cname)

        self.cat_cols = sorted(cat_cols)
        self.num_cols = sorted(num_cols)

    def init_key_info(self):
        if self.num_multi_idx == 0:
            self.key_col = None
            self.key_set = None
            self.key_num = 1
        elif self.num_multi_idx == 1:
            self.key_col = self.id_cols[0]
            self.key_set = set(self.train_X[self.key_col])
            self.key_num = len(self.key_set)
        else:
            log('create a unique key in KeyCross')

    def init_col2type(self):
        for col in self.cat_cols:
            self.col2type[col] = CONSTANT.CATEGORY_TYPE
        for col in self.num_cols:
            self.col2type[col] = CONSTANT.NUMERICAL_TYPE

    def add_post_drop_col(self, col):
        self.post_drop_set.add(col)

    def add_cross_cols(self, col):
        self.cross_cols.add(col)




