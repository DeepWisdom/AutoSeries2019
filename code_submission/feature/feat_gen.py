import gc
from log_utils import log, timeclass
import CONSTANT
from joblib import Parallel, delayed
from .feat_opt import *
from .feat_namer import FeatNamer
from data_utils import downcast, check_density, revert_pivot_feat_join, numpy_downward_fill
from functools import partial
import time


class KeysCross:
    #@timeclass(cls='KeysCross')
    def fit(self, table):
        pass

    #@timeclass(cls='KeysCross')
    def transform(self, table, mode='pred'):
        if table.num_multi_idx < 2:
            return

        col2type = {}
        todo_cols = table.id_cols
        opt = c_cats_combine
        if mode == 'train':
            df = table.train_X
            todo_cols = table.id_cols
            for col in todo_cols:
                mx = df[col].max()
                table.col2max[col] = mx
            col2type = {}
            obj = '(' + ')('.join(todo_cols) + ')'
            param = None
            new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.CATEGORY_TYPE)
            col2type[new_col] = CONSTANT.CATEGORY_TYPE

            key_ss = opt(df[todo_cols], table.col2max)
            df[new_col] = key_ss

            table.key_col = new_col
            table.key_set = set(key_ss)
            table.key_num = len(table.key_set)
            log(f'{self.__class__.__name__} pipeline update key info, key_col: {table.key_col}, key_num: {table.key_num}')
        elif mode == 'pred':
            df = table.pred_X
            key_ss = opt(df[todo_cols], col2max=table.col2max)
            df[table.key_col] = key_ss

        table.update_data(df, col2type, mode)

        del key_ss
        gc.collect()

    #@timeclass(cls='KeysCross')
    def fit_transform(self, table):
        self.fit(table)
        self.transform(table, mode='train')
        
        
class KeysTimeCross:
    #@timeclass(cls='KeysTimeCross')
    def fit(self, table):
        self.col_name = None

    #@timeclass(cls='KeysTimeCross')
    def transform(self, table, mode='pred'):
        if table.num_multi_idx < 1:
            return

        col2type = {}
        todo_cols = [table.key_col] + [col for col in table.time_attr_cols if ('hour' in col)or ('minute' in col)]
        opt = c_cats_combine
        if mode == 'train':
            df = table.train_X
            for col in todo_cols:
                mx = df[col].max() + 1
                table.col2max[col] = mx
            col2type = {}
            obj = '(' + ')('.join(todo_cols) + ')'
            param = None
            new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.CATEGORY_TYPE)
            col2type[new_col] = CONSTANT.CATEGORY_TYPE
            self.col_name = new_col

            key_ss = opt(df[todo_cols], table.col2max)
            df[new_col] = key_ss

        elif mode == 'pred':
            df = table.pred_X
            key_ss = opt(df[todo_cols], col2max=table.col2max)
            df[self.col_name] = key_ss

        table.update_data(df, col2type, mode)

        if mode == 'train':
            table.add_post_drop_col(new_col)
            table.add_cross_cols(new_col)

        del key_ss
        gc.collect()

    #@timeclass(cls='KeysTimeCross')
    def fit_transform(self, table):
        self.fit(table)
        self.transform(table, mode='train')


class LagFeat:
    def __init__(self):
        self.windows = (3, 7, 14, 30)

    #@timeclass(cls='LagFeat')
    def fit(self, table):
        pass

    #@timeclass(cls='LagFeat')
    def transform(self, table):
        df = table.pred_X
        # log(f'before transform df')
        # print(df.head())
        todo_cols = [table.label]
        col2idx = table.col2idx
        if table.is_multivar:
            keys = df[table.key_col].values
            idxs = table.keys2idxs(keys)
            hist = table.key_window
            for window in self.windows:
                if type(window) == int:
                    rolled = hist[idxs, :, -window:]
                    window_size = window
                else:
                    start, end = window
                    rolled = hist[idxs, :, -end:-start]
                    window_size = end - start
                lag_mean = np.mean(rolled, axis=2)
                lag_std = np.std(rolled, axis=2) * np.sqrt(window_size/(window_size-1))
                lag_max = np.max(rolled, axis=2)
                lag_min = np.min(rolled, axis=2)
                for col in todo_cols:
                    idx = col2idx[col]
                    df[f'{col}_lag_{window}_mean'] = lag_mean[:, idx]
                    df[f'{col}_lag_{window}_std'] = lag_std[:, idx]
                    df[f'{col}_lag_{window}_max'] = lag_max[:, idx]
                    df[f'{col}_lag_{window}_min'] = lag_min[:, idx]
                    #df[f'{col}_lag_1'] = hist[idxs, :, -1]
        else:
            hist = table.key_window
            for window in self.windows:
                if type(window) == int:
                    rolled = hist[0, :, -window:]
                    window_size = window
                else:
                    start, end = window
                    rolled = hist[0, :, -end:-start]
                    window_size = end - start
                lag_mean = np.mean(rolled, axis=1)
                lag_std = np.std(rolled, axis=1) * np.sqrt(window_size/(window_size-1))
                lag_max = np.max(rolled, axis=1)
                lag_min = np.min(rolled, axis=1)
                for col in todo_cols:
                    idx = col2idx[col]
                    df[f'{col}_lag_{window}_mean'] = lag_mean[idx]
                    df[f'{col}_lag_{window}_std'] = lag_std[idx]
                    df[f'{col}_lag_{window}_max'] = lag_max[idx]
                    df[f'{col}_lag_{window}_min'] = lag_min[idx]
        # import pdb;pdb.set_trace()

        # log('lag transformed df')
        # print(df.head(10))

    #@timeclass(cls='LagFeat')
    def fit_transform(self, table):
        todo_cols = [table.label]
        if not todo_cols:
            return

        df = table.train_X
        # log(f'before fit_trans df')
        # print(df.head())
        col2type = {}

        opt = n_lag
        opt = partial(opt, todo_cols=todo_cols, windows=self.windows)

        if table.is_multivar:
            #if not CONSTANT.USE_TIME_STAMP_MODE:
            if table.key_num < CONSTANT.KEY_NUM:
                log('use order')
                groups = df[[table.key_time_col] + [table.key_col] + todo_cols].groupby(table.key_col)
                groups = [group[1] for group in groups]
                #res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(opt)(group) for group in groups)
                res = []
                for group in groups:
                    tmp = opt(group)
                    res.append(tmp)
                if res:
                    tmp = pd.concat(res)
                    tmp = tmp.drop(todo_cols, axis=1)
                    for col in tmp.columns:
                        if col != table.key_col and col != table.key_time_col:
                            col2type[col] = CONSTANT.NUMERICAL_TYPE
                    print(f'before merge shape: {df.shape}')
                    df = pd.merge(df, tmp, how='left', on=[table.key_time_col, table.key_col])
                    print(f'after merge shape: {df.shape}')
                    del tmp
                    gc.collect()
            else:
                time_col = table.key_time_col
                cat_col = table.key_col

                new_cols = []
                for window in self.windows:
                    for col in todo_cols:
                        series_matrix = pd.pivot_table(df, index=[time_col], values=[col],
                                                       columns=cat_col,)# aggfunc=[np.mean])
                        series_matrix = series_matrix.fillna(method="ffill")
                        series_matrix_shift1 = series_matrix.shift(1)

                        lag_mean = series_matrix_shift1.rolling(window).mean()
                        lag_std = series_matrix_shift1.rolling(window).std()
                        lag_max = series_matrix_shift1.rolling(window).max()
                        lag_min = series_matrix_shift1.rolling(window).min()

                        new_col = f'{col}_lag_{window}_mean'
                        col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                        new_cols.append(new_col)
                        df = revert_pivot_feat_join(df, lag_mean, new_col)

                        new_col = f'{col}_lag_{window}_std'
                        col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                        new_cols.append(new_col)
                        df = revert_pivot_feat_join(df, lag_std, new_col)

                        new_col = f'{col}_lag_{window}_max'
                        col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                        new_cols.append(new_col)
                        df = revert_pivot_feat_join(df, lag_max, new_col)

                        new_col = f'{col}_lag_{window}_min'
                        col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                        new_cols.append(new_col)
                        df = revert_pivot_feat_join(df, lag_min, new_col)

        else:
            tmp = opt(df[[table.key_time_col]+todo_cols])
            tmp.drop([table.key_time_col] + todo_cols, axis=1, inplace=True)
            for col in tmp.columns:
                col2type[col] = CONSTANT.NUMERICAL_TYPE
                df[col] = tmp[col]
            del tmp
            gc.collect()
        table.update_data(df, col2type, mode='train')
        #log(f'{self.__class__.__name__} produce {len(col2type)} features')

        # log('lag fir_transformed df')
        # print(df.head(10))

class WindowEncode:
    def __init__(self):
        self.windows = (3,)

    @timeclass(cls='WindowEncode')
    def fit(self, table):
        pass

    @timeclass(cls='WindowEncode')
    def transform(self, table):
        df = table.pred_X
        # log(f'before transform df')
        # print(df.head())
        cat_cols = table.id_cols
        todo_cols = table.init_num_cols + [table.label]
        col2idx = table.col2idx

        cat2idxs = {}
        for cat in cat_cols:
            objs = set(df[cat])
            idxs = table.objs2idxs(objs, cat)
            cat2idxs[cat] = idxs

        cat2window = table.cat2window

        def window_encode(cat, cat_ss, hist, idxs, num_cols, windows):
            cat_set = set(cat_ss)
            cat_list = list(cat_set)
            res = []
            for window in windows:
                rolled = hist[idxs, :, -window:]
                lag_mean = np.mean(rolled, axis=2)
                lag_max = np.max(rolled, axis=2)
                lag_min = np.min(rolled, axis=2)
                for col in num_cols:
                    idx = col2idx[col]
                    mean_ss = lag_mean[:, idx]
                    max_ss = lag_max[:, idx]
                    min_ss = lag_min[:, idx]
                    mean_map = {cat_list[i]: mean_ss[i] for i in range(len(cat_list))}
                    max_map = {cat_list[i]: max_ss[i] for i in range(len(cat_list))}
                    min_map = {cat_list[i]: min_ss[i] for i in range(len(cat_list))}
                    new_mean_ss = cat_ss.map(mean_map)
                    new_max_ss = cat_ss.map(max_map)
                    new_min_ss = cat_ss.map(min_map)
                    new_mean_ss.name = f'{col}_{cat}_lag_{window}_mean'
                    new_max_ss.name = f'{col}_{cat}_lag_{window}_max'
                    new_min_ss.name = f'{col}_{cat}_lag_{window}_min'
                    res.extend([new_mean_ss, new_max_ss, new_min_ss])
            new_df = pd.concat(res, axis=1)
            return new_df

        opt = window_encode
        opt = partial(opt, num_cols=todo_cols, windows=self.windows)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')\
            (delayed(opt)(cat, df[cat], cat2window[cat], cat2idxs[cat]) for cat in cat_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            for col in tmp.columns:
                df[col] = tmp[col]
            del tmp
            gc.collect()

        # log('lag transformed df')
        # print(df.head(10))

    @timeclass(cls='WindowEncode')
    def fit_transform(self, table):
        cat_cols = table.id_cols
        todo_cols = table.init_num_cols + [table.label]
        key_time_col = table.key_time_col

        df = table.train_X

        col2type = {}

        # def window_encode(df, cat, time, todo_cols, windows):
        #     groups = df[[time] + [cat] + todo_cols].groupby(cat)
        #     groups = [group[1] for group in groups]
        #     res = []
        #     for group_df in groups:
        #         pre_lag = 1
        #         for window in windows:
        #             rolled = group_df[todo_cols].shift(pre_lag).rolling(window=window)
        #             group_df = group_df.join(rolled.mean().add_suffix(f'_{cat}_lag_{window}_mean'))
        #             group_df = group_df.join(rolled.max().add_suffix(f'_{cat}_lag_{window}_max'))
        #             group_df = group_df.join(rolled.min().add_suffix(f'_{cat}_lag_{window}_min'))
        #         res.append(group_df)
        #     tmp = pd.concat(res)
        #     #tmp.fillna(method='bfill', inplace=True)
        #     tmp.drop(todo_cols+[time, cat], axis=1, inplace=True)
        #     return tmp

        def window_encode(df, cat, time, todo_cols, windows):
            group_df = df[[time] + [cat] + todo_cols].groupby([cat, time]).mean()
            res = []
            pre_lag = 1
            for window in windows:
                rolled = group_df[todo_cols].shift(pre_lag).rolling(window=window)
                group_df = group_df.join(rolled.mean().add_suffix(f'_{cat}_lag_{window}_mean'))
                group_df = group_df.join(rolled.max().add_suffix(f'_{cat}_lag_{window}_max'))
                group_df = group_df.join(rolled.min().add_suffix(f'_{cat}_lag_{window}_min'))
            res.append(group_df)
            tmp = pd.concat(res, axis=1)
            tmp.reset_index(drop=False, inplace=True)
            tmp.drop(todo_cols, axis=1, inplace=True)
            print(f'shape before merge{df.shape}')
            tmp = pd.merge(df[[time, cat]], tmp, how='left', on=[time, cat])
            print(f'shape after merge{tmp.shape}')
            #tmp.fillna(method='bfill', inplace=True)
            tmp.drop([time, cat], axis=1, inplace=True)
            return tmp

        opt = window_encode
        opt = partial(opt, time=key_time_col, todo_cols=todo_cols, windows=self.windows)

        # for cat in cat_cols:
        #     tmp = opt(df, cat)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(opt)(df, cat) for cat in cat_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            for col in tmp.columns:
                col2type[col] = CONSTANT.NUMERICAL_TYPE
                df[col] = tmp[col]
            del tmp
            gc.collect()
            table.update_data(df, col2type, 'train')
            log(f'{self.__class__.__name__} produce {len(col2type)} features')


class TimeDate:

    #@timeclass(cls='TimeDate')
    def fit(self, table):
        df = table.train_X
        todo_col = table.key_time_col
        self.attrs = []
        for atr in ['year', 'month', 'day', 'hour', 'weekday']:
            atr_ss = getattr(df[todo_col].dt, atr)
            if atr_ss.nunique() > 1:
                self.attrs.append(atr)

    #@timeclass(cls='TimeDate')
    def transform(self, table, mode='pred'):
        todo_col = table.key_time_col
        if mode == 'train':
            df = table.train_X
        elif mode == 'pred':
            df = table.pred_X
        col2type = {}

        new_cols = []
        opt = time_atr
        for atr in self.attrs:
            obj = todo_col
            param = atr
            new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.CATEGORY_TYPE)
            new_cols.append(new_col)
            col2type[new_col] = CONSTANT.CATEGORY_TYPE
            df[new_col] = opt(df[todo_col], atr)

        ts = df[todo_col]
        ts = pd.to_datetime(ts, unit='s')
        param = 'timestamp'
        new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
        df[new_col] = ts.astype('int64') // 10 ** 9
        new_cols.append(new_col)
        table.time_attr_cols = new_cols
        col2type[new_col] = CONSTANT.NUMERICAL_TYPE

        table.update_data(df, col2type, mode)
        #log(f'{self.__class__.__name__} produce {len(new_cols)} features')

    #@timeclass(cls='TimeDate')
    def fit_transform(self, table):
        self.fit(table)
        self.transform(table, mode='train')


class GroupMean:
    def __init__(self):
        self.res = {}
        self.cols1 = []

    @timeclass(cls='GroupMean')
    def fit(self, table):
        #self.cols1 = table.cat_cols
        self.cols1 = ['c_TimeDate:A1:hour']
        key_col = table.key_col

        cols2 = [table.label]
        if len(self.cols1) == 0 or len(cols2) == 0:
            return
        df = table.train_X

        df['key_cross'] = df['c_TimeDate:A1:hour'] * df[key_col]
        self.cols1 = ['key_cross']

        for col1 in self.cols1:
            for col2 in cols2:
                obj = f'({col1})({col2})'
                param = None
                new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)

                mean_ss = df.groupby([col1], sort=False)[col2].mean()
                mean_ss = downcast(mean_ss)
                self.res[new_col] = mean_ss.to_dict()

    @timeclass(cls='GroupMean')
    def transform(self, table, mode='pred'):
        cols1 = self.cols1
        cols2 = [table.label]
        if len(cols1) == 0 or len(cols2) == 0:
            return
        if mode == 'train':
            df = table.train_X
        elif mode == 'pred':
            df = table.pred_X

        key_col = table.key_col
        df['key_cross'] = df['c_TimeDate:A1:hour'] * df[key_col]

        new_cols = []
        col2type = {}
        for col1 in cols1:
            for col2 in cols2:
                obj = f'({col1})({col2})'
                param = None
                new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
                new_cols.append(new_col)
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                key_ss = df[col1]
                if new_col in self.res:
                    mean_dict = self.res[new_col]
                    feat_ss = key_ss.map(mean_dict)
                    # nan_ratio = feat_ss.isna().mean()
                    # if nan_ratio > 0:
                    #     df[new_col] = df.groupby([col1])[col2].mean()
                    # else:
                    #     df[new_col] = feat_ss
                    df[new_col] = feat_ss
        table.update_data(df, col2type, mode)
        if mode == 'train':
            log(f'{self.__class__.__name__} produce {len(new_cols)} features')

    @timeclass(cls='GroupMean')
    def fit_transform(self, table):
        self.fit(table)
        self.transform(table, mode='train')


class WindowMinus:
    def __init__(self):
        self.windows = (3, 7)

    def fit(self, tabel):
        pass

    def transform(self, table, mode='pred'):
        if mode == 'train':
            df = table.train_X
        elif mode == 'pred':
            df = table.pred_X

        col2type = {}
        todo_cols = [table.label]
        window_num = len(self.windows)
        for col in todo_cols:
            for i in range(window_num):
                for j in range(i, window_num):
                    win1 = self.windows[i]
                    win2 = self.windows[j]
                    for opt in ['mean', 'std']:
                        col1 = f'{col}_lag_{win1}_{opt}'
                        col2 = f'{col}_lag_{win2}_{opt}'
                        total_cols = df.columns
                        if col1 in total_cols and col2 in total_cols:
                            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                            col2type[f'{col1}_minus_{col2}'] = CONSTANT.NUMERICAL_TYPE

        table.update_data(df, col2type, mode)
        if mode == 'train':
            log(f'{self.__class__.__name__} produce {len(col2type)} features')

    def fit_transform(self, table):
        self.transform(table, mode='train')


class Delta:
    def __init__(self):
        self.cols_name = {}

    def fit(self, tabel):
        pass
    #
    # def update_feature_matrix(self, new_hist):
    #     print(self.series_matrix)

    def transform(self, table, mode='pred'):
        df = table.pred_X
        todo_cols = [table.label]
        col2idx = table.col2idx
        if table.is_multivar:
            keys = df[table.key_col].values
            idxs = table.keys2idxs(keys)
            hist = table.key_window
            for col in todo_cols:
                idx = col2idx[col]
                df[self.cols_name['delta']] = hist[idxs, idx, -1] - hist[idxs, idx, -2]
                df[self.cols_name['delta_delta']] = - (hist[idxs, idx, -2] - hist[idxs, idx, -3])\
                                                    + (hist[idxs, idx, -1] - hist[idxs, idx, -2])
                df[self.cols_name['ratio']] = (hist[idxs, idx, -1] - hist[idxs, idx, -2]) / (hist[idxs, idx, -2] + 1e-3)
        else:
            hist = table.key_window
            for col in todo_cols:
                idx = col2idx[col]
                df[self.cols_name['delta']] = hist[0, idx, -1] - hist[0, idx, -2]
                df[self.cols_name['delta_delta']] = - (hist[0, idx, -2] - hist[0, idx, -3])\
                                                    + (hist[0, idx, -1] - hist[0, idx, -2])
                df[self.cols_name['ratio']] = (hist[0, idx, -1] - hist[0, idx, -2]) / (hist[0, idx, -2] + 1e-3)

    def fit_transform(self, table):
        df = table.train_X

        time_col = table.key_time_col
        num_col = table.label
        cat_col = table.key_col
        col2type = {}

        if table.key_num < CONSTANT.KEY_NUM:
            log('use order')
            def n_diff(df, time_col, cat_col, num_col):
                index = df.index
                df.reset_index(drop=True, inplace=True)
                if cat_col is not None:
                    df.sort_values([cat_col, time_col], inplace=True)
                else:
                    df.sort_values([time_col], inplace=True)

                num_ss = df[num_col]

                delta = num_ss.diff()
                if cat_col is not None:
                    cat_ss = df[cat_col].diff()
                    cat_ss2 = df[cat_col].diff(2)

                delta_ratio = delta / (num_ss.shift(1) + 1e-3)

                delta = delta.shift(1)
                delta_ratio = delta_ratio.shift(1)
                delta_delta = delta.diff()

                delta = downcast(delta)
                delta_delta = downcast(delta_delta)
                delta_ratio = downcast(delta_ratio)

                if cat_col is not None:
                    delta[cat_ss != 0] = np.nan
                    delta_ratio[cat_ss != 0] = np.nan
                    delta_delta[cat_ss2 != 0] = np.nan

                new_df = pd.concat([delta, delta_delta, delta_ratio], axis=1)

                new_df.sort_index(inplace=True)
                new_df.index = index

                return new_df

            new_cols = []
            obj = f'({cat_col})({num_col})'
            param = 'delta'
            new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
            self.cols_name['delta'] = new_col
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            new_cols.append(new_col)

            obj = f'({cat_col})({num_col})'
            param = 'delta_delta'
            new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
            self.cols_name['delta_delta'] = new_col
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            new_cols.append(new_col)

            obj = f'({cat_col})({num_col})'
            param = 'ratio'
            new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
            self.cols_name['ratio'] = new_col
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            new_cols.append(new_col)

            opt = n_diff
            opt = partial(opt, time_col=time_col, num_col=num_col, cat_col=cat_col)

            if cat_col is not None:
                tmp = opt(df[[time_col, num_col, cat_col]])
            else:
                tmp = opt(df[[time_col, num_col]])

            tmp.columns = new_cols
            for col in new_cols:
                df[col] = tmp[col]

            table.update_data(df, col2type, mode='train')
            del tmp
            gc.collect()
        else:
            self.series_matrix = pd.pivot_table(df, index=[time_col], values=[num_col], columns=cat_col, aggfunc=[np.mean])
            self.series_matrix = self.series_matrix.fillna(method="ffill")

            series_matrix_1 = self.series_matrix.shift(1)
            series_matrix_2 = self.series_matrix.shift(2)
            # series_matrix_3 = series_matrix.shift(3)
            series_matrix_delta_1 = series_matrix_1 - series_matrix_2
            # series_matrix_delta_2 = series_matrix_2 - series_matrix_3
            # series_matrix_deltadelta_1 = series_matrix_delta_1 - series_matrix_delta_2
            series_matrix_deltadelta_1 = series_matrix_delta_1.diff()

            series_matrix_diffrto_1 = series_matrix_delta_1 / (series_matrix_2 + 1e-3)

            new_cols = []
            obj = f'({cat_col})({num_col})'
            param = 'delta'
            new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
            self.cols_name['delta'] = new_col
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            new_cols.append(new_col)
            df = revert_pivot_feat_join(df, series_matrix_delta_1, new_col)

            obj = f'({cat_col})({num_col})'
            param = 'delta_delta'
            new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
            self.cols_name['delta_delta'] = new_col
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            new_cols.append(new_col)
            df = revert_pivot_feat_join(df, series_matrix_deltadelta_1, new_col)

            obj = f'({cat_col})({num_col})'
            param = 'ratio'
            new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
            self.cols_name['ratio'] = new_col
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            new_cols.append(new_col)
            df = revert_pivot_feat_join(df, series_matrix_diffrto_1, new_col)

            table.update_data(df, col2type, mode='train')



# class IsNewKey:
#     def fit(self, tabel):
#         pass
#
#     def transform(self, table, mode='pred'):
#         pass
#
#     def fit_transform(self, table):
#         if not table.is_multivar:
#             return
#
#         df = table.train_X
#         key_col = table.key_col
#
#         col2type = {}
#
#         obj = f'({key_col})'
#         param = None
#         new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
#         col2type[new_col] = CONSTANT.NUMERICAL_TYPE
#
#         idxs = df.groupby(key_col).first().index
#         df[new_col] = 0
#         df.loc[idxs] = 1
#
#         import pdb;pdb.set_trace()
#
#         table.update_data(df, col2type, mode='train')



# class TimeSinceLast:
#     def fit(self, tabel):
#         pass
#
#     def transform(self, table, mode='pred'):
#         df = table.pred_X
#         # log(f'before transform df')
#         # print(df.head())
#         todo_cols = [table.time_attr_cols[-1]]
#         col2idx = table.col2idx
#         if table.is_multivar:
#             keys = df[table.key_col].values
#             idxs = table.keys2idxs(keys)
#             hist = table.key_window
#             for window in self.windows:
#                 rolled = hist[idxs, :, -window:]
#                 lag_mean = np.mean(rolled, axis=2)
#                 lag_std = np.std(rolled, axis=2) * np.sqrt(window/(window-1))
#                 lag_max = np.max(rolled, axis=2)
#                 lag_min = np.min(rolled, axis=2)
#                 for col in todo_cols:
#                     idx = col2idx[col]
#                     df[f'{col}_lag_{window}_mean'] = lag_mean[:, idx]
#                     df[f'{col}_lag_{window}_std'] = lag_std[:, idx]
#                     df[f'{col}_lag_{window}_max'] = lag_max[:, idx]
#                     df[f'{col}_lag_{window}_min'] = lag_min[:, idx]
#                     #df[f'{col}_lag_1'] = hist[idxs, :, -1]
#         else:
#             hist = table.key_window
#             for window in self.windows:
#                 rolled = hist[0, :, -window:]
#                 lag_mean = np.mean(rolled, axis=1)
#                 lag_std = np.std(rolled, axis=1) * np.sqrt(window/(window-1))
#                 lag_max = np.max(rolled, axis=1)
#                 lag_min = np.min(rolled, axis=1)
#                 for col in todo_cols:
#                     idx = col2idx[col]
#                     df[f'{col}_lag_{window}_mean'] = lag_mean[idx]
#                     df[f'{col}_lag_{window}_std'] = lag_std[idx]
#                     df[f'{col}_lag_{window}_max'] = lag_max[idx]
#                     df[f'{col}_lag_{window}_min'] = lag_min[idx]
#
#     def fit_transform(self, table):
#         if not table.is_multivar:
#             return
#
#         df = table.train_X
#         key_col = table.key_col
#         time_col = table.time_attr_cols[-1]
#
#         col2type = {}
#
#         obj = f'({key_col})({time_col})'
#         param = '-1'
#         new_col = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)
#         col2type[new_col] = CONSTANT.CATEGORY_TYPE
#
#         df[new_col] = df.groupby(key_col)[time_col].diff().fillna(0)
#
#         table.update_data(df, col2type, mode='train')





