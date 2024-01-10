import etna
import numpy as np
import pandas as pd
import typing as tp

from scipy.stats.stats import pearsonr
from sklearn.linear_model import OrthogonalMatchingPursuit

from etna.auto import Tune
from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE
from etna.models import LinearPerSegmentModel, ElasticPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import (
    LagTransform, LogTransform, DifferencingTransform, MeanTransform,
    STLTransform
)
from etna.transforms.decomposition.deseasonal import DeseasonalityTransform
from etna.transforms.math.power import YeoJohnsonTransform
from etna.transforms.missing_values.imputation import TimeSeriesImputerTransform


# import etna_utils
from . import load_data_utils
from . import prepare_data

# import granger_utils

from .transforms import TsIntegrateTransform

N_FOLDS = 55


def select_long_series(
    other_dfs_all_d: dict,
    print_failed: bool = False
) -> list[tp.Tuple[str, str, str]]:

    info_list = []
    for names, tser in load_data_utils.iter_series(other_dfs_all_d):
        try:
            if len(tser.dropna()) < 21 or tser.index.min() >= pd.to_datetime('2016-01-01'):
                continue
            info_list.append(names)
        except (ValueError, TypeError):
            if print_failed:
                print('failed', names)

    return info_list


def prepare_ts_all(
    ser: pd.Series,
    other_dfs_all_d: dict,
    info_dict,
    exog_df_shift=0,
) -> etna.datasets.tsdataset.TSDataset:
    ts_many = prepare_data.prepare_multivariate_dataset(
        ser,
        other_dfs_all_d,
        info_dict,
        exog_df_shift=exog_df_shift
    )

    start_date = ser.dropna().index.min()
    ts_many = TSDataset(
        ts_many.raw_df[start_date:],
        freq='MS',
        df_exog=ts_many.df_exog[start_date:],
        known_future=[]
    )

    return ts_many


def prepare_ts_subset(
    ser: pd.Series,
    other_dfs_all: dict,
    not_bad_names: list,
    last_train_date = pd.to_datetime('2019-01-01')
) -> TSDataset:
    ts = prepare_data.prepare_multivariate_dataset(
        ser,
        other_dfs_all,
        not_bad_names
    )
    add_ser_to_ts(ts, ser.rename('integrated'), ser.name)
    startd_date = ser.dropna().index.min()
    ts = etna.datasets.tsdataset.TSDataset(
        ts.raw_df[startd_date:last_train_date],
        freq='MS',
        df_exog=ts.df_exog[startd_date: last_train_date],
        known_future=[]
    )
    return ts

def add_ser_to_ts(ts, ser, segment) -> None:
    df = ser.to_frame().reset_index(names='timestamp')
    df['segment'] = segment
    df = TSDataset.to_dataset(df)

    ts.add_columns_from_pandas(df, update_exog=True)


# select vars to preidct

# def run_granger_tests(
#     ser_des: pd.Series,
#     other_dfs_all_d: dict
# ) -> dict:
#     def corr(x: pd.Series, y: pd.Series):
#         df = x.to_frame().join(y).dropna()
#         x, y = df.iloc[:, 0], df.iloc[:, 1]
#         return pearsonr(x, y)[0]
    
#     info_dict = dict()
#     for names, tser in etna_utils.load_data_utils.iter_series(other_dfs_all_d):
#         try:
#             if len(tser.dropna()) < 21 or tser.index.min() >= pd.to_datetime('2016-01-01'):
#                 continue
#             res = granger_utils.granger_test(ser_des, tser.shift(1), maxlag=5, verbose=0)
#             pval = granger_utils.get_min_pval(res, maxlag=5)
#             info_dict[names] = (pval, corr(ser_des, tser))
#         except (ValueError, TypeError):
#             print('failed', names)
    
#     info_dict = dict(sorted(info_dict.items(), key=lambda item: -np.abs(item[1][1])))

#     return info_dict

def prepare_transforms(
        info_dict: dict, horizon, prepare_integrated: bool = False, tr_lags: int=3
) -> list:
    if prepare_integrated:
        transforms = [
            TsIntegrateTransform(in_column='integrated'),
	        LogTransform(in_column='integrated'),
	        STLTransform(in_column='integrated', period=12, robust=True,
                    #   model_kwargs=dict(order=(1, 1, 1))
					stl_kwargs=dict(
                        seasonal=15,
                        trend=13
                    )
					# model='arima'
            ),
            MeanTransform(in_column='integrated', window=4, alpha=0.9, out_column='integrated_mean'),
            YeoJohnsonTransform(in_column='integrated_mean', standardize=True),

			LagTransform(
	            in_column='integrated_mean',
	            lags=[horizon],
	            out_column='integrated_mean_lag'
	        ),
            # YeoJohnsonTransform(in_column='integrated', standardize=True)
        ]
    else:
        transforms = []

    for col in info_dict:
        # print(col)
        transforms.extend([
            TimeSeriesImputerTransform(in_column=col[2], strategy='forward_fill'),
            TimeSeriesImputerTransform(in_column=col[2], strategy='mean'),
	        STLTransform(in_column=col[2], period=12, robust=True),
            MeanTransform(in_column=col[2], window=3, out_column=f'{col[2]}_mean'),
            LagTransform(
                in_column=f'{col[2]}_mean',
                lags=[horizon],
            ),
            YeoJohnsonTransform(in_column=f'{col[2]}_mean', standardize=True)
        ])
    
    return transforms


def remove_columns_many_nas(
        df: pd.DataFrame,
        na_pers: int = 50,
    ) -> pd.DataFrame:
    to_remove = []

    for c in df:
        if df[c].isna().mean() > na_pers / 100:
            to_remove.append(c)

    return df.drop(columns=to_remove)


def choose_vars_omp(
    ser: pd.Series,
    ts_many: etna.datasets.tsdataset.TSDataset,
    omp_n_nonzero: int = 15
):
    # prepare data
    tdf = ts_many.to_pandas()[ser.name]
    y_df = tdf['target']
    x_df = tdf.copy()
    del x_df['target']
    mask = y_df.isna()
    y_df = y_df.loc[~mask]
    x_df = x_df.loc[~mask]
    x_df = remove_columns_many_nas(x_df)
    x_df = x_df.ffill()

    mask = x_df.isna().any(axis=1)
    x_df, y_df = map(lambda df: df[~mask], [x_df, y_df])

    # x_df = x_df.dropna()

    # x_df = x_df.dropna(axis=1)

    # select
    model = OrthogonalMatchingPursuit(
        n_nonzero_coefs=omp_n_nonzero,
        normalize=True
    )
    model.fit(x_df, y_df)
    
    coef_dict = dict(zip(model.feature_names_in_, model.coef_))
    non_zero = {k: v for k, v in coef_dict.items() if v != 0}
    non_zero = dict(sorted(non_zero.items(), key=lambda item: -np.abs(item[1])))

    return non_zero
    

def _find_names(other_dfs_all_d: dict, non_zero, use_eval: bool = False):
    class Finder:
        def __init__(self, other_dfs_all_d=other_dfs_all_d, use_eval=use_eval):
            names = list(k for k, _ in load_data_utils.iter_others(other_dfs_all_d))
            self.map = dict(zip([n[2] for n in names], names))
            self.use_eval = use_eval

        def find(self, name: str):
            if 'integrated' in name.lower() or 'target' in name.lower():
                return name

            if use_eval:
                return self.map[eval(name).in_column]
            try:
                return self.map[name]
            except KeyError:
                print('failed', name)
    

    finder = Finder(other_dfs_all_d, use_eval=use_eval)
    not_bad_names = list({finder.find(n) for n in non_zero})

    return not_bad_names

# fit 


def tune_selected_vars(
    ser: pd.Series,
    other_dfs_all: dict,
    not_bad_names: list,
    horizon: int,
    n_folds: int = 30,
    last_train_date = pd.to_datetime('2019-01-01'),
    n_trials=3,

    return_full_res: bool = False
) -> Pipeline:
    # ts = etna_utils.prepare_data.prepare_multivariate_dataset(
    #     ser,
    #     other_dfs_all,
    #     not_bad_names
    # )
    # add_ser_to_ts(ts, ser.rename('integrated'), ser.name)
    # ts = etna.datasets.tsdataset.TSDataset(
    #     ts[:last_train_date],
    #     freq='MS'
    # )

    ts = prepare_ts_subset(
        ser,
        other_dfs_all,
        not_bad_names,
        last_train_date=last_train_date
    )

    transformes = prepare_transforms(not_bad_names, prepare_integrated=True)
    transformes.extend([
        DeseasonalityTransform(in_column='target', period=12),
        YeoJohnsonTransform(in_column='target', standardize=True),
        MeanTransform(in_column='target', window=3, alpha=0.75, out_column='target_mean'),
        LagTransform(
            in_column='target_mean',
            lags=1,
            out_column='target_mean_lag'
        ),
        LagTransform(
            in_column='target',
            lags=[horizon, horizon + 1],
            out_column='target_lag'
        )
    ])

    # model = LinearPerSegmentModel()
    model = ElasticPerSegmentModel()

    pipe = Pipeline(
        model = model,
        transforms=transformes,
        horizon=horizon
    )

    p2t = pipe.params_to_tune()
    for k, v in p2t.copy().items():
        if 'robust' in k or 'strategy' in k:
            del p2t[k]
        if 'multiplicative' in str(v):
            del p2t[k]
        if 'per-segment' in str(v):
            del p2t[k]

    tune = Tune(
        pipeline=pipe,
        target_metric=MAE(),
        horizon=horizon,
        backtest_params=dict(n_folds=n_folds, joblib_params=dict(verbose=0)),
        params_to_tune=p2t
    )

    best_pipe = tune.fit(ts=ts, n_trials=n_trials)
    if return_full_res:
        return tune.top_k(k=1)[0], tune, best_pipe

    return tune.top_k(k=1)[0]


def backtest_good_pipe(
    ser: pd.Series,
    other_dfs_all: dict,
    not_bad_names: list,
    best_pipe: Pipeline, 
    last_train_date = None,
    n_folds=N_FOLDS
) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if last_train_date is None:
        last_train_date = ser.index.max()
    
    ts = prepare_ts_subset(
        ser,
        other_dfs_all,
        not_bad_names,
        last_train_date=last_train_date
    )

    m_df, f_df, fold_info_df = best_pipe.backtest(
        ts = ts,
        metrics=[MAE()],
        n_folds=n_folds,
        joblib_params=dict(verbose=0),
        mode = 'constant'
    )

    return m_df, f_df, fold_info_df


def choose_good_vars_names_omp(
        ser: pd.Series,
        other_dfs_all_d: dict,
        omp_n_nonzero = 15
    ) -> tp.List[tp.Tuple[str, str, str]]:
    names_list = select_long_series(other_dfs_all_d)
    
    # choose vars
    ts_all = prepare_ts_all(
        ser,
        other_dfs_all_d,
        names_list
    )
    not_bad_names = choose_vars_omp(ser, ts_all, omp_n_nonzero=omp_n_nonzero)

    not_bad_names = _find_names(other_dfs_all_d, not_bad_names)
    
    return not_bad_names


def get_metrics_selected_strategy(
    ser: pd.Series,
    other_dfs_all: dict,
    other_dfs_all_d: dict,
    horizon: int = 1,
    omp_n_nonzero = 15,
    n_trials = 3
    
):
    # choose vars
    not_bad_names = choose_good_vars_names_omp(ser, other_dfs_all_d, omp_n_nonzero)

    # auto tune
    pipe, tune, pipe_dict = tune_selected_vars(
        ser,
        other_dfs_all,
        not_bad_names,
        horizon,
        return_full_res=True,
        n_trials=n_trials
    )

    # run backtest
    m_df, f_df, fold_info_df = backtest_good_pipe(
        ser, other_dfs_all, not_bad_names, pipe
    )

    return tune, not_bad_names, m_df, f_df, fold_info_df
