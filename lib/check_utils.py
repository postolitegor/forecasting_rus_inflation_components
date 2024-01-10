import typing as tp
# from wsgiref.util import request_uri

import numpy as np
import pandas as pd
import pickle
# import plotly.graph_objects as go
# import statsmodels.api as sm

from copy import deepcopy
from dataclasses import dataclass
from sklearn.metrics import (
    mean_absolute_error as MAE
    , mean_squared_error as MSE
)

from pathlib import Path
from tqdm import tqdm

from etna.datasets.tsdataset import TSDataset
from etna.pipeline import Pipeline
from etna.transforms import TimeSeriesImputerTransform

# from .etna_utils import load_data_utils
from etna_utils import forecast_utils


PROCESS_DATA_FOLDER = Path(__file__).resolve().parent / 'processed_data'
HORIZONS_ALL = [1, 2, 3, 6, 12, 24]
TRAIN_END_DATE = pd.to_datetime('2018-12-01')

# =========
# laspeires functions
# =========

def prepare_weights(
        cpi_df: pd.DataFrame,
        init_weights_df: pd.DataFrame,
):
    
    def get_price_df(inf_df):
        weights_df = (1 + inf_df / 100).cumprod()
        weights_df = weights_df.shift()
        weights_df.fillna(1, inplace=True)
        return weights_df
    
    weights_df = cpi_df.groupby(by=lambda ind: ind.year).apply(get_price_df)
    weights_df = weights_df * init_weights_df

    weights_df = weights_df.apply(lambda row: row / np.nansum(row), axis=1)
    return weights_df


def agg_laspeires(cpi_df, init_weights_df):

    weights_df = prepare_weights(cpi_df, init_weights_df)
    # res = (cpi_mom500_df[cols500] * weights_df).sum(axis=1)
    res = (cpi_df * weights_df).sum(axis=1)
    return res


# =========
# load data
# =========

@dataclass
class ForecastData:
    target_name: str
    forecast_df: pd.DataFrame
    metrics_df: pd.DataFrame
    fold_info_df: pd.DataFrame
    pipeline_info: dict
    path: str
    agg_mae: tp.Optional[float] = None


def load_datas_all() -> tp.List[ForecastData]:
    with open(PROCESS_DATA_FOLDER / 'datas (all).pickle', 'rb') as f:
        datas = pickle.load(f)
    return datas


def select_by_model_name(
    datas,
    model_types: tp.Iterable[str] = ('ARDPerSegmentModel', 'ElasticPerSegmentModel')
) -> tp.List[ForecastData]:
    ans = []
    for d in datas:
        if d.pipeline_info['model_name'] in model_types:
            ans.append(d)
    return ans


def load_datas_selected(
    model_types: tp.Iterable[str] = ('ARDPerSegmentModel', 'ElasticPerSegmentModel')
) -> tp.List[ForecastData]:
    datas = load_datas_all()

    datas = select_by_model_name(datas)
    return datas
      

# ===========
# select data
# ===========


def _prepare_forecast_df(
    d: ForecastData,
    ser_true: tp.Optional[pd.Series] = None,
    drop_dates: tp.Optional[tp.Tuple] = None,
    last_month_only: bool = False,
    use_head: bool = True
) -> pd.DataFrame:
    horizon = d.pipeline_info['horizon']
    target_name = d.pipeline_info['target_name']
    # target_name = prev2new_names.get(target_name, target_name)
    forecast_df = d.forecast_df[target_name][['fold_number', 'target']]
    fold_info_df = d.fold_info_df
    # ser_true = cpi_mom_full_df[target_name]


    if d.pipeline_info['model_name'] in ('ARDPerSegmentModel', 'ElasticPerSegmentModel'):
        f_df = forecast_df.groupby('fold_number').head(1)
       
        # if horizon == 1:

    if horizon == 1:
        f_df = forecast_df

    elif horizon == 2:
        if use_head:
            f_df = forecast_df.groupby('fold_number').head(1)
        else:
            f_df = forecast_df.groupby('fold_number').tail(1)
        
    elif horizon == 3:
        if use_head:
            f_df = forecast_df.groupby('fold_number').head(1)
        else:
            f_df = forecast_df.groupby('fold_number').tail(1)

    elif horizon == 6:
        n = 1 if last_month_only else 3    
        if use_head:
            f_df = forecast_df.groupby('fold_number').head(n)
        else:
            f_df = forecast_df.groupby('fold_number').tail(n)

    elif horizon == 12:
        n = 1 if last_month_only else 6     
        if use_head:
            f_df = forecast_df.groupby('fold_number').tail(n)
        else:
            f_df = forecast_df.groupby('fold_number').tail(n)

    elif horizon == 24:
        n = 1 if last_month_only else 12
        if use_head:
            f_df = forecast_df.groupby('fold_number').tail(n)
        else:
            f_df = forecast_df.groupby('fold_number').tail(n)

    elif horizon == 36:
        n = 1 if last_month_only else 12
        if use_head:
            f_df = forecast_df.groupby('fold_number').tail(n)
        else:
            f_df = forecast_df.groupby('fold_number').tail(n)
    
    if ser_true is not None:
        f_df = f_df.join(ser_true)
    
    if drop_dates is not None:
        mask = (drop_dates[0] <= f_df.index) & (f_df.index < drop_dates[1])
        f_df = f_df[~mask]

    return f_df

def _calc_mae(some_forecast_df: pd.DataFrame) -> float:
    err = np.abs(some_forecast_df['target'].astype(float) -\
        some_forecast_df[some_forecast_df.columns[-1]].astype(float))
    return np.nanmean(err)

def _calc_rmse(some_forecast_df: pd.DataFrame) -> float:
    err = np.abs(some_forecast_df['target'].astype(float) -\
        some_forecast_df[some_forecast_df.columns[-1]].astype(float))
    return np.sqrt(np.nanmean(err ** 2))

def _calc_std(some_forecast_df: pd.DataFrame) -> float:
    err = np.abs(some_forecast_df['target'].astype(float) -\
        some_forecast_df[some_forecast_df.columns[-1]].astype(float))
    return np.nanstd(err)


# def get_av_mae(
#     # horizon: int,
#     # forecast_df: pd.DataFrame,
#     # fold_info_df: pd.DataFrame,
#     d: ForecastData,
#     ser_true: pd.Series,
#     drop_dates: tp.Optional[tp.Tuple] = None
# ) -> float:
#     df = _prepare_forecast_df(d. ser_true, drop_dates)

#     mae = 

#     err = np.abs(df['target'].astype(float) - df[ser_true.name].astype(float))
#     return err.mean()


def datas2pd_all(
        datas,
        dates_without: tp.Tuple,
        cpi_mom_full_df,
        last_month_only: bool = False
    ) -> pd.DataFrame:
    """

    Args:
        datas (_type_):

    Returns:
        pd.DataFrame: with columns:
            horizon, category_name, model_name, mae(full), mae (with dates excluded), comment
    """
    l2pd = [None] * len(datas)
    for i, d in tqdm(enumerate(datas), total=len(datas)):
        if d.target_name in ('other food', 'other non-food', 'other services'):
            continue

        # skip d with d.stride != 1
        if (d.fold_info_df['train_end_time'][1] - d.fold_info_df['train_end_time'][0]).days > 31:
            continue

        try:
            cpi_mom_full_df[d.target_name]
            true_col_name = d.target_name
        except KeyError:
            true_col_name = d.target_name.replace('_', ':')

        forc_df = _prepare_forecast_df(d, cpi_mom_full_df[true_col_name], None, last_month_only=last_month_only)
        forc_df_cut = _prepare_forecast_df(d, cpi_mom_full_df[true_col_name], dates_without, last_month_only=last_month_only)

        mae = _calc_mae(forc_df)
        mae_cut = _calc_mae(forc_df_cut)
        rmse = _calc_rmse(forc_df)
        rmse_cut = _calc_rmse(forc_df_cut)

        l2pd[i] = {
            'horizon': d.pipeline_info['horizon'],
            'category_name': d.pipeline_info['target_name'],
            'model_name': d.pipeline_info['model_name'],
            # 'mae (full)': d.metrics_df['MAE'].mean(),
            'mae (full)': mae,
            f'mae without [{dates_without[0].strftime("%Y.%m.%d")}-{dates_without[1].strftime("%Y.%m.%d")})': mae_cut,
            'rmse (full)': rmse,
            f'rmse without [{dates_without[0].strftime("%Y.%m.%d")}-{dates_without[1].strftime("%Y.%m.%d")})': rmse_cut, 
            'comment': d.pipeline_info['comment'],
            'path': d.path
        }
    # return l2pd

    idx_to_drop = []
    for i, dd in enumerate(l2pd):
        if dd is None:
            idx_to_drop.append(i)
    
    for i in reversed(idx_to_drop):
        l2pd.pop(i)

    # l2pd = [
    #     {
    #         'horizon': d.pipeline_info['horizon'],
    #         'category_name': d.pipeline_info['target_name'],
    #         'model_name': d.pipeline_info['model_name'],
    #         # 'mae (full)': d.metrics_df['MAE'].mean(),
    #         'mae (full)': get_av_mae(d, cpi_mom_full_df[d.pipeline_info['target_name']]),
    #         f'mae without [{dates_without[0]}-{dates_without[1]})':
    #             get_av_mae(d, cpi_mom_full_df[d.pipeline_info['target_name']], dates_without),
    #         'rmse (full)': ,
    #         'rmse (without) [{dates_without[0]}-{dates_without[1]})': , 
    #         'comment': d.pipeline_info['comment']
    #     } for d in tqdm(datas) if not (d.target_name == 'other food' or d.target_name == 'other non-food' or d.target_name == 'other services')
    # ]

    return pd.DataFrame(l2pd)


def get_best_model_by_av_mae(
    stats_df: pd.DataFrame,
    horizon: int,
    sort_category_name = 'mae without [2022.02.01-2022.05.01)'
    ):
    stats_df = stats_df[stats_df['horizon'] == horizon]

    def func(df: pd.DataFrame):
        return df.sort_values(by=sort_category_name).head(1)

    stats_df = stats_df.groupby(by='category_name').apply(func)

    ans = dict(zip(stats_df['category_name'], stats_df['path']))

    return ans


def get_best_forecasts(
    datas: tp.List[ForecastData],
    horizon: int,
    stats_df: pd.DataFrame,
    cols: tp.List[str],
    sort_category_name = 'mae without [2022.02.01-2022.05.01)'
):
    datas_map = dict()

    for d in datas:
        datas_map[d.path] = d

    best_dict = get_best_model_by_av_mae(stats_df, horizon, sort_category_name)
    
    ans = pd.DataFrame()

    for col in cols:
        d = datas_map[best_dict[col]]
        df = _prepare_forecast_df(d)
        ans[col] = df['target'].rename(col)
    
    return ans


# long forecast period

def select_datas(
    # datas: tp.List[ForecastData],
    category_name: str,
    stats_df: pd.DataFrame,
    sort_category_name: str = 'rmse (without) [2022.02.01-2022.05.01)'
):
    # best_models_df = metrics_best_by_category_df[metrics_best_by_category_df['category_name'] == category_name]
    def func(df, top_n = 1):
        return df.sort_values(by=sort_category_name).head(top_n)
    
    horizon2best_model_name = dict()
    horizon2path = dict()

    for h in HORIZONS_ALL:
        tdf = stats_df[(stats_df['category_name'] ==  category_name) & (stats_df['horizon'] == h)][['horizon', 'model_name', sort_category_name, 'path']]\
            .groupby('model_name', group_keys=True).apply(func).sort_values(by=sort_category_name).head(1)
        horizon2path[h] = tdf['path'].values[0]
        horizon2best_model_name[h] = tdf['model_name'].values[0]
    
    return horizon2path


def get_forecast_prev_df(
    d: ForecastData,
    horizon: int
) -> pd.DataFrame:
    horizon2num_tail = {
        1: 1,
        2: 1,
        3: 1,
        6: 3,
        12: 6,
        24: 12
    }
    num_tail = horizon2num_tail[horizon]

    forc_df = d.forecast_df[d.target_name][['fold_number', 'target']].groupby('fold_number').tail(num_tail)
    fold_info_df = d.fold_info_df
    forc_df['fold_number'] = forc_df['fold_number'].astype(int)
    fold_info_df['fold_number'] = fold_info_df['fold_number'].astype(int)
    
    f_df = pd.merge(
        forc_df.reset_index(),
        fold_info_df[['fold_number', 'train_end_time', 'test_start_time', 'test_end_time']],
        how='outer',
        on='fold_number'
    )

    f_df['horizon'] = horizon
    del f_df['fold_number']
    f_df.rename(columns={'timestamp': 'forecast_date', 'index': 'forecast_date'}, inplace=True)
    f_df = f_df[f_df['train_end_time'] >= TRAIN_END_DATE]

    return f_df


def get_forecast_final_df(
    d: ForecastData,
    horizon: int
) -> pd.DataFrame:
    horizon2num_tail = {
        1: 1,
        2: 1,
        3: 1,
        6: 3,
        12: 6,
        24: 12
    }
    num_tail = horizon2num_tail[horizon]
    forc_df = d.forecast_df[d.target_name][['fold_number', 'target']].groupby('fold_number').head(num_tail)
    fold_info_df = d.fold_info_df
    forc_df['fold_number'] = forc_df['fold_number'].astype(int)
    fold_info_df['fold_number'] = fold_info_df['fold_number'].astype(int)


get_forecast_final_df = get_forecast_prev_df


def get_forecasts_long_horizons(
    cat_name: str,
    stats_df: pd.DataFrame,
    horizon2path: tp.Dict[int, str],
    path2data: tp.Dict[str, ForecastData],
    max_horizon: int = 24
):
    horizon2path = select_datas(cat_name, stats_df)
        
    f_dfs = []
    if max_horizon >= 1:
        path = horizon2path[1]
        d = path2data[path]
        if 'final' in d.pipeline_info['comment']:
            f_df1 = get_forecast_final_df(d, 1)
        else:
            f_df1 = get_forecast_prev_df(d, 1)

        f_dfs.append(f_df1)

    if max_horizon >= 2:
        path = horizon2path[2]
        d = path2data[path]
        if 'final' in d.pipeline_info['comment']:
            f_df2 = get_forecast_final_df(d, 2)
        else:
            f_df2 = get_forecast_prev_df(d, 2)

        f_dfs.append(f_df2)

    if max_horizon >= 3:
        path = horizon2path[3]
        d = path2data[path]

        if 'final' in d.pipeline_info['comment']:
            f_df3 = get_forecast_final_df(d, 3)
        else:
            f_df3 = get_forecast_prev_df(d, 3)

        f_dfs.append(f_df3)


    if max_horizon >= 6:
        path = horizon2path[6]
        d = path2data[path]
        if 'final' in d.pipeline_info['comment']:
            f_df6 = get_forecast_final_df(d, 6)
        else:
            f_df6 = get_forecast_prev_df(d, 6)

        f_dfs.append(f_df6)


    if max_horizon >= 12:
        path = horizon2path[12]
        d = path2data[path]
        if 'final' in d.pipeline_info['comment']:
            f_df12 = get_forecast_final_df(d, 12)
        else:
            f_df12 = get_forecast_prev_df(d, 12)

        f_dfs.append(f_df12)


    if max_horizon >= 24:
        path = horizon2path[24]
        d = path2data[path]
        if 'final' in d.pipeline_info['comment']:
            f_df24 = get_forecast_final_df(d, 24)
        else:
            f_df24 = get_forecast_prev_df(d, 24)

        f_dfs.append(f_df24)

    return pd.concat(f_dfs).sort_values(by=['train_end_time', 'forecast_date'])


def get_forecasts_long(
    cat_name: str,
    stats_df: pd.DataFrame,
    path2data: tp.Dict[str, ForecastData]
) -> pd.DataFrame:
    horizon2path = select_datas(cat_name, stats_df)
    df = get_forecasts_long_horizons(cat_name, stats_df, horizon2path, path2data)

    return df



# for in-sample forecasts


def read_from_folder(folder: Path) -> ForecastData:
    with open(folder / 'forecast.pkl', 'rb') as f:
        forecast_df = pickle.load(f)

    with open(folder / 'metrics.pkl', 'rb') as f:
        metrics_df = pickle.load(f)

    with open(folder / 'info.json', 'r') as f:
        pipeline_info = json.load(f)

    with open(folder / 'fold_inf.pkl', 'rb') as f:
        fold_info_df = pickle.load(f)

    target_name = pipeline_info['target_name']

    return ForecastData(
        target_name,
        forecast_df,
        metrics_df,
        fold_info_df,
        pipeline_info,
        str(folder)
    )

def select_pipes(
    #...
    stats_good_df: pd.DataFrame,
    select_by: str = 'mae without [2022.02.01-2022.05.01)',
) -> tp.Dict[tp.Tuple[str, int], str]:
    def func(df):
        return df.sort_values(by=select_by).head(1)
    selected_df = stats_good_df.groupby(by=['category_name', 'horizon']).apply(func)
    ans = dict()
    for idx, p in zip(selected_df['path'].index, selected_df['path']):
        name = idx[0]
        horizon = idx[1]
        ans[(name, horizon)] = p
    
    return ans


def load_pipe(folder: Path) -> Pipeline:
    with open(folder / 'pipe.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


def load_pipes(
    pipes_paths: tp.Iterable[str]
):
    res = (load_pipe(path) for path in pipes_paths)
    return res


def select_vars(
    pipe: Pipeline,
    other_dfs_all
) -> tp.List[tp.Tuple[str, str, str]]:
    def collect_other_names(p: Pipeline):
        names = []
        for t in p.transforms:
            if t.in_column not in ('integrated', 'target'):
                if isinstance(t.in_column, str):
                    names.append(t.in_column)
                elif isinstance(t.in_column, list):
                    names.extend(t.in_column)
                else:
                    raise NotImplementedError
        f_names = []
        f_names = [n for n in names if not n.endswith('_mean') and n not in ('integrated', 'target')]
        return list(set(f_names))

    names = collect_other_names(pipe)
    names_triplets = forecast_utils._find_names(other_dfs_all, names)

    return names_triplets


def _impute_ts(ts: TSDataset) -> TSDataset:
    cols = [c[1] for c in ts.df_exog.columns]

    trs = []
    for c in cols:
        trs.append(
            TimeSeriesImputerTransform(in_column=c, strategy='forward_fill')
        )

    ts_copy = deepcopy(ts)
    ts_copy.fit_transform(trs)
    return ts_copy


def run(
    pipe: Pipeline,
    ts: TSDataset
) -> pd.DataFrame:
    forecast_ts = pipe.forecast()
    pass


def save(
    path: str,
    forecasts_df: pd.DataFrame
):
    pass