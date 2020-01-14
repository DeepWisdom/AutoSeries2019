import pandas as pd
import numpy as np
from data_utils import parse_time
from log_utils import log
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)
pd.set_option('max_colwidth', 100)

class AutoEDA:

    def label_dist(self, table):
        X = table.train_X
        time_ss = X[table.time_col]
        label_ss = X[table.label]
        num_cols = table.init_num_cols
        id_cols = table.id_cols

        todo_cols = [table.label]

        df = parse_time(time_ss)
        df[table.label] = label_ss
        for col in num_cols:
            df[col] = X[col]
        for col in id_cols:
            df[col] = X[col]

        df_year = pd.pivot_table(df, index=['year'], values=todo_cols, aggfunc=[np.mean, np.std])
        log('label distribute by year')
        print(df_year)

        df_month = pd.pivot_table(df, index=['month'], values=todo_cols, aggfunc=[np.mean, np.std])
        log('label distribute by month')
        print(df_month)

        df_weekday = pd.pivot_table(df, index=['weekday'], values=todo_cols, aggfunc=[np.mean, np.std])
        log('label distribute by weekday')
        print(df_weekday)

        df_day = pd.pivot_table(df, index=['day'], values=todo_cols, aggfunc=[np.mean, np.std])
        log('label distribute by day')
        print(df_day)

        df_year_month = pd.pivot_table(df, index=['year', 'month'], values=todo_cols, aggfunc=[np.mean, np.std])
        log('label distribute by year-month')
        print(df_year_month)

        if 'hour' in df.columns:
            df_hour = pd.pivot_table(df, index=['hour'], values=todo_cols, aggfunc=[np.mean, np.std])
            log('label distribute by hour')
            print(df_hour)

            df_year_hour = pd.pivot_table(df, index=['year', 'hour'], values=todo_cols, aggfunc=[np.mean, np.std])
            log('label distribute by year-hour')
            print(df_year_hour)

            df_month_hour = pd.pivot_table(df, index=['month', 'hour'], values=todo_cols, aggfunc=[np.mean, np.std])
            log('label distribute by month-hour')
            print(df_month_hour)

            df_day_hour = pd.pivot_table(df, index=['day', 'hour'], values=todo_cols, aggfunc=[np.mean, np.std])
            log('label distribute by day-hour')
            print(df_day_hour)

        df_key = pd.pivot_table(df, index=['year', 'month', 'day']+table.id_cols, values=todo_cols, aggfunc=[np.mean, np.std])
        log('lable distribute by year-month-day-key')
        print(df_key)

    def data_info(self, df):

        log(f'train data shape: {df.shape}')

        log('describe data')
        print(df.describe())

        log('nan ratio')
        print(df.isna().mean())

    def get_nan_col(self, df):
        ss = df.isna().mean()

        drop_nan_col = list(ss[ss>0.6].index)
        log(f'drop col with high nan ratio: {drop_nan_col}')
        df.drop(drop_nan_col, axis=1, inplace=True)

        fill_nan_col = list(ss[(ss > 0) & (ss <= 0.6)].index)

        log(f'cols need fill value: {fill_nan_col}')

        return fill_nan_col







