import pickle
import pandas as pd
import numpy as np
from models.lgb import LGBMRegressor
# from old_model import LGBMRegressor
from models.rg import RidgeModel
from models.lasso import LassoModel
import os
import gc
from table import Table
from transform import CatTransformer, NumTransformer
from feature.feat_engine import FeatEngine
from feature .feat_pipeline import FeatPipeline
from autoeda import AutoEDA
from log_utils import log, timeit
import time
import math
from sklearn.metrics import mean_squared_error
import CONSTANT
from data_utils import trans_label, inverse_label, get_rmse

#os.system("pip install -i https://pypi.tuna.tsinghua.edu.cn/simple hyperopt")


class Model:
    def __init__(self, info, test_timestamp, pred_timestamp):
        self.info = info
        self.primary_timestamp = info['primary_timestamp']
        self.primary_id = info['primary_id']
        self.label = info['label']
        self.schema = info['schema']

        #print(f"\ninfo: {self.info}")

        self.origin_cat_cols = [col for col, types in self.schema.items() if types == 'str']
        self.origin_num_cols = [col for col, types in self.schema.items() if types == 'num' and col != self.label]

        self.nan_num_cols = None

        self.test_timestamp = test_timestamp
        self.pred_timestamp = pred_timestamp

        self.n_test_timestamp = len(pred_timestamp)
        self.update_interval = None

        #log(f"sample of test record: {len(test_timestamp)}")
        #log(f"number of pred timestamp: {len(pred_timestamp)}")

        self.best_model = None

        self.n_predict = -1
        self.n_predict_true = 0
        self.n_update = 0
        self.preds = None

        self.rg_scores = []
        self.lgb_scores = []
        self.lasso_scores = []
        self.ensemble_scores = []
        self.scores = []

        self.rg_recent_score = 1
        self.lgb_recent_score = 1
        self.lasso_recent_score = 1

        self.n_train = 0

        self.cat2cat_trans = {}
        self.num2num_trans = {}

        self.models = {}

        self.models['lgb'] = {}
        self.models['rg'] = {}
        self.models['lasso'] = {}

        for name, loader in self.models.items():
            loader['score'] = []
            loader['cnt'] = 0

        self.models['lgb']['model'] = LGBMRegressor()
        self.models['rg']['model'] = RidgeModel()
        self.models['lasso']['model'] = LassoModel()

        self.table = None

        self.autoeda = AutoEDA()

        self.hist_window = []
        self.trans_log = False

        self.train_duration = None

        self.model = None

        self.explore_duration = None

        self.explore_interval = None

        self.valid_mode = False

        #print(f"Finish init\n")

    def preprocess(self, df, mode='fit_trans'):
        if mode == 'fit_trans':
            for cat in self.origin_cat_cols:
                self.cat2cat_trans[cat] = CatTransformer()
                df[cat] = self.cat2cat_trans[cat].fit_transform(df[cat])
            # for num in self.origin_num_cols:
            #     self.num2num_trans[num] = NumTransformer()
            #     df[num] = self.num2num_trans[num].fit_transform(df[num])
        elif mode == 'trans':
            for cat in self.origin_cat_cols:
                df[cat] = self.cat2cat_trans[cat].fit_transform(df[cat])
            # for num in self.origin_num_cols:
            #     df[num] = self.num2num_trans[num].fit_transform(df[num])

    def train(self, train_data, time_info):
        t0 = time.time()
        print(f"\nTrain time budget: {time_info['train']}s")
        self.n_train += 1
        if self.n_train == 1:
            X = train_data
            shape1 = X.shape
            print(f'shape1: {shape1}')
            X.drop_duplicates(keep='first', inplace=True)
            shape2 = X.shape
            print(f'shape1: {shape2}')
            self.preprocess(X, mode='fit_trans')
            self.table = Table(X, self.info)

        # feat generate
        feat_pipline = FeatPipeline()
        feat_engine = FeatEngine(feat_pipline)
        feat_engine.fit_transform_oder1s(self.table)
        self.feat_engine = feat_engine

        X, y, cat = self.table.fit_transform_output()

        t1 = time.time()
        if self.n_update == 0:
            do_explore = True
        else:
            do_explore = False
        time_left = time_info['train'] - (t1 - t0)
        self.explore_space(X, y, cat, time_left, do_explore=do_explore)
        t2 = time.time()

        drop = []
        for name, loader in self.models.items():
            try:
                loader['model'].fit(X, y, cat)
            except:
                drop.append(name)

        for name in drop:
            self.models.pop(name)

        if self.table.key_window is None:
            self.table.init_key_window()

        # release cache
        self.table.train_X = None
        self.table.train_y = None

        print("Finish train\n")

        next_step = 'predict'

        self.explore_duration = t2 - t1

        if self.train_duration is None:
            self.train_duration = time.time() - t0 - self.explore_duration

        max_update_num = int(time_info['update']/self.train_duration) + 1
        self.update_interval = (self.n_test_timestamp - self.n_predict_true) / max_update_num

        return next_step

    @timeit
    def predict(self, new_history, pred_record, time_info):
        t0 = time.time()

        if len(self.models) == 0:
            preds = np.zeros(len(pred_record))
            return preds, 'predict'

        self.n_predict += 1
        self.n_predict_true += 1
        print(self.n_predict_true)
        if self.n_predict > 0:
            self.preprocess(new_history, mode='trans')
            self.table.add_key_window(new_history)

        if self.preds is not None:
            last_label = new_history[self.label].values
            for name, loader in self.models.items():
                model_mean_squar = mean_squared_error(last_label, loader['preds'])
                loader['score'].append(model_mean_squar)
                loader['recent_rmse'] = np.mean(loader['score'][-3:])
            # real_mean_squar = mean_squared_error(last_label, self.preds)
            # self.scores.append(real_mean_squar)
            # move_score = math.sqrt(np.mean(self.scores))
            # print(f'mean score{move_score}')

        self.preprocess(pred_record, mode='trans')

        #pred_record = self.fill_na(pred_record, mode='pred')
        self.table.pred_X = pred_record

        self.feat_engine.transform_oder1s(self.table)

        pred_X = self.table.transform_output()

        coef = max(len(self.models)-1, 1)
        fm = 0

        for name, loader in self.models.items():
            model = loader['model']
            loader['preds'] = model.predict(pred_X)
            fm += loader['recent_rmse']

        self.preds = 0
        drop_list = []
        if len(self.models) > 1:
            for name, loader in self.models.items():
                weight = (fm - loader['recent_rmse'])/(coef * fm)
                log(f'{name}: {weight}')
                self.preds += weight * loader['preds']
                if weight < 0.05:
                    loader['cnt'] += 1
                if loader['cnt'] > 10:
                    drop_list.append(name)
        else:
            for name, loader in self.models.items():
                self.preds = loader['preds']
        for name in drop_list:
            self.models.pop(name)
            log(f'drop model {name}')

        if time_info['update'] < (self.train_duration*2) or self.n_predict_true > self.n_test_timestamp-3:
            log('No time for update')
            do_update = False
        else:
            do_update = True

        if self.n_predict > self.update_interval and do_update:
            next_step = 'update'

            self.n_predict = 0
            self.n_update += 1
        else:
            next_step = 'predict'

        predict_time = time.time() - t0

        if predict_time * (self.n_test_timestamp - self.n_predict_true) > time_info['predict']:
            max_rmse = 0
            for name, loader in self.models.items():
                if loader['recent_rmse'] > max_rmse:
                    drop_name = name
                    max_rmse = loader['recent_rmse']
            self.models.pop(drop_name)

        return list(self.preds), next_step

    def update(self, train_data, test_history_data, time_info):
        t0 = time.time()
        total_data = pd.concat([train_data, test_history_data])
        total_data.reset_index(drop=True, inplace=True)

        self.n_train = 0
        self.train(total_data, time_info)

        next_step = 'predict'

        self.train_duration = time.time() - t0 - self.explore_duration
        return next_step

    def save(self, model_dir, time_info):
        print(f"\nSave time budget: {time_info['save']}s")

        pkl_list = []
        for attr in dir(self):
            if attr.startswith('__') or attr in ['train', 'predict', 'update', 'save', 'load', 'preprocess', 'fill_na']:
                continue

            pkl_list.append(attr)
            pickle.dump(getattr(self, attr), open(os.path.join(model_dir, f'{attr}.pkl'), 'wb'))

        pickle.dump(pkl_list, open(os.path.join(model_dir, f'pkl_list.pkl'), 'wb'))

        print("Finish save\n")

    def load(self, model_dir, time_info):
        print(f"\nLoad time budget: {time_info['load']}s")

        pkl_list = pickle.load(open(os.path.join(model_dir, 'pkl_list.pkl'), 'rb'))

        for attr in pkl_list:
            setattr(self, attr, pickle.load(open(os.path.join(model_dir, f'{attr}.pkl'), 'rb')))

        print("Finish load\n")

    def explore_space(self, X, y, cat, time_left, do_explore):
        drop = []
        model2time = {}
        if do_explore:
            for name, loader in self.models.items():
                t1 = time.time()
                model = loader['model']
                print(time_left)
                if name == 'lasso':
                    if 'rg' in model2time:
                        if time_left < (model2time['rg'] + 40):
                            drop.append(name)
                            continue
                try:
                    rmse = model.explore_params(X, y, cat)
                except:
                    drop.append(name)
                    t2 = time.time()
                    model2time[name] = t2 - t1
                    time_left = time_left - (t2 - t1)
                    continue
                loader['recent_rmse'] = rmse
                t2 = time.time()
                model2time[name] = t2 - t1
                time_left = time_left - (t2 - t1)
                log(f'{name} rmse: {rmse} time {t2-t1}')

            for name in drop:
                self.models.pop(name)
                log(f'model {name} error')
















