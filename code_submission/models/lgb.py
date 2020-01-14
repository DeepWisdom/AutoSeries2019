import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
import CONSTANT
from log_utils import log, timeclass
import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from data_utils import get_rmse, discret
import autosample
pd.set_option('display.max_rows', 200)


class LGBMRegressor:
    def __init__(self):
        self.model = None
        self.rg_bad_cols = None


        self.params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            #'num_class': 93,
            "metric": "rmse",
            "verbosity": -1,
            "seed": CONSTANT.SEED,
            "num_threads": CONSTANT.JOBS,
        }

        self.hyperparams = {
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 255,
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            "learning_rate": 0.02,
        }

        self.oder1s_params_space = {'num_leaves': np.arange(30, 61, 10),
                                    'max_depth': [-1, 2, 4, 6, 8, 10],
                                    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
                                    # 'lambda_l1': [ 0.2,  0.4,  0.6],
                                    # 'lambda_l2': [ 0.2,  0.4,  0.6]}

    #@timeclass(cls='LGBMRegressor')
    def fit(self, X, y, categories):

        lgb_train = lgb.Dataset(X, y)

        self.model = lgb.train({**self.params, **self.hyperparams}, train_set=lgb_train)

        return self

    #@timeclass(cls='LGBMRegressor')
    def predict(self, X_test):

        return self.model.predict(X_test)

    @timeclass(cls='LGBMRegressor')
    def random_search(self, X, y, categories):
        lgb_clf = lgb.LGBMRegressor(n_jobs=CONSTANT.JOBS, boosting_type='gbdt', n_estimators=300,
                                    random_state=1)
                                    #categorical_feature=categories)

        gs = RandomizedSearchCV(estimator=lgb_clf,
                                param_distributions=self.oder1s_params_space,
                                scoring='neg_mean_squared_error',
                                cv=TimeSeriesSplit(n_splits=3),
                                refit=True,
                                random_state=1,
                                verbose=True)

        gs.fit(X, y)

        log(f'cv result is {gs.cv_results_}')
        #self.params_bs['params'].update(gs.best_params_)
        self.hyperparams.update(gs.best_params_)
        log(f'best params:{gs.best_params_}')

    @timeclass(cls='LGBMRegressor')
    def early_stop_opt(self, X_train, X_eval, y_train, y_eval, categories):

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        model = lgb.train({**self.params, **self.hyperparams}, verbose_eval=20, num_boost_round=1000,
                          train_set=lgb_train, valid_sets=lgb_eval, valid_names='eval',
                          early_stopping_rounds=50) #categorical_feature=categories)
        # model = lgb.train(**self.params_bs, verbose_eval=20,
        #                   train_set=lgb_train, valid_sets=lgb_eval, valid_names='eval',
        #                   early_stopping_rounds=50,)# categorical_feature=categories)

        self.params['num_boost_round'] = model.best_iteration
        #self.params_bs['num_boost_round'] = model.best_iteration
        log(f'best boost round: {model.best_iteration}')

    def log_feat_importances(self):
        importances = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                                    'importances': self.model.feature_importance("gain")})

        importances.sort_values('importances', ascending=False, inplace=True)

        log('feat importance:')
        log(f'{importances.head(200)}')

        return list(importances['features'][:30])

    @timeclass(cls='LGBMRegressor')
    def bayes_opt(self, X_train, X_eval, y_train, y_eval, categories):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_eval, label=y_eval)

        params = self.params
        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
            "max_depth": hp.choice("max_depth", [-1, 5, 7, 9]),
            "num_leaves": hp.choice("num_leaves", np.linspace(20, 61, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "min_child_weight": hp.uniform('min_child_weight', 0.01, 1),
            "min_split_gain": hp.uniform('min_split_gain', 0.001, 0.1),
            "min_data_in_leaf": hp.choice("min_data_in_leaf", np.linspace(50, 100, 10, dtype=int))
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 300, valid_data,
                              #categorical_feature=categories,
                              early_stopping_rounds=50, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            # in classification, less is better
            return {'loss': score, 'status': STATUS_OK}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=30, verbose=1,
                             rstate=np.random.RandomState(1))
        self.hyperparams.update(space_eval(space, best))
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {self.hyperparams}")

    def explore_params(self, X, y, categories):
        num_cols = [col for col in X.columns]

        X = X.iloc[-100000:]
        y = y.iloc[-100000:]

        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=0)

        self.bayes_opt(X_train, X_eval, y_train, y_eval, categories=categories)
        #self.random_search(X_train, y_train, categories=categories)
        self.early_stop_opt(X_train, X_eval, y_train, y_eval, categories=categories)


        lgb_train = lgb.Dataset(X_train, y_train)

        #self.model = lgb.train(**self.params_bs, train_set=lgb_train,)# categorical_feature=categories)
        self.model = lgb.train({**self.params, **self.hyperparams}, train_set=lgb_train, num_boost_round=1000)
        preds = self.model.predict(X_eval)

        rmse = get_rmse(preds, y_eval)
        log(f'valid rmse: {rmse}')

        imp_cols = self.log_feat_importances()

        return rmse






