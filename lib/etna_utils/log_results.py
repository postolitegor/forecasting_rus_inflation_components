import json
import numpy as np
import os
import pandas as pd
import pickle
import plotly
import sklearn
import sys
import typing as tp

from pathlib import Path, PosixPath

import etna


RESULTS_PATH = Path().absolute() / 'results'


# ============================
# functions for pipeline2dict
# ============================

def calc_last_month_error(
        ts: etna.datasets.tsdataset.TSDataset,
        forecast_df: pd.DataFrame,
        col_name: str,
        metric: tp.Callable = sklearn.metrics.mean_absolute_error
) -> float:
    """calc. last month forecast mae

    Args:
        ts (etna.datasets.tsdataset.TSDataset):
        forecast_df (pd.DataFrame):
        col_name (str):
        metric (tp.Callable, optional):

    Returns:
        float: last month forecast
    """
    fdf = forecast_df[col_name].groupby('fold_number').tail(1)
    all_df = pd.merge(
        fdf['target'].rename('forecast'),
        ts.to_pandas()[col_name]['target'],
        left_index=True,
        right_index=True
    )

    return metric(all_df['target'], all_df['forecast'])



def get_class_name(obj):
    return type(obj).__name__


def get_attrs(obj):
    try:
        return str(obj.__dict__) #  some adapters are note json serializable
    except:
        return str(obj)
    

def get_transforms(model):
    return str(model)


def get_n_folds(pipe):
    return pipe.n


def pipeline2dict(
        ts: etna.datasets.tsdataset.TSDataset,
        pipe: etna.pipeline.Pipeline,
        n_folds: tp.Union[int, tp.List[etna.pipeline.FoldMask]],
        forecast_df: pd.DataFrame,
        target_name: str,
        comment: str = ''

) -> dict:
    """Convert pipeline to json format
    Ideally, one should be able to restore everything given this json (if possible)

    Args:
        ts (etna.datasets.tsdataset.TSDataset): ts on which pipeline was trained
        pipe (etna.pipeline.Pipeline): 
        n_folds (tp.Union[int, tp.List[etna.pipeline.FoldMask]]):
        forecast_df (pd.DataFrame): result from backtest function
        target_name (str):
        comment (str, optional): any comment you would like to store

    Returns:
        dict:
    """
    assert isinstance(target_name, str)
    res = {
        'horizon': pipe.horizon,
        'model_name': get_class_name(pipe.model),
        'model_params': get_attrs(pipe.model),
        'model_transforms': get_transforms(pipe.transforms),
        'n_folds': n_folds,
        'comment': comment,
        'average_mae': calc_last_month_error(ts, forecast_df, target_name),
        'target_name': target_name,
        'other_vars': '' #TODO: add vars from TSDataset
        }
    
    return res


# ==============================
# save figure functions
# ==============================

def get_last_month_forecast(
        forecast_df: pd.DataFrame,
        col_name: str
) -> pd.DataFrame:
    """returns forecast for last month only (used for drawing forecast)

    Args:
        forecast_df (pd.DataFrame):
        col_name (str):

    Returns:
        pd.DataFrame:
    """
    columns = forecast_df.columns
    res_df = forecast_df[col_name].groupby('fold_number').tail(1)
    res_df.columns = columns
    return res_df


def get_figure(
        ts: etna.datasets.tsdataset.TSDataset,
        forecast_df: pd.DataFrame,
        col_name: str
) -> plotly.graph_objects.Figure:
    """returns plotly figure of forecasts

    Args:
        ts (etna.datasets.tsdataset.TSDataset):
        forecast_df (pd.DataFrame):
        col_name (str):

    Returns:
        _type_:
    """
    last_forecast_df = get_last_month_forecast(forecast_df, col_name)
    fig = etna.analysis.plot_backtest_interactive(last_forecast_df, ts)
    return fig


# ===============================================
# files and folders utilities
# ===============================================

def _default_info_dict2_folder_name(info_dict: dict) -> str:
    """return name of an "inner folder" for logging

    Args:
        info_dict (dict):

    Returns:
        str:
    """
    return 'h' + str(info_dict['horizon']) + '_mae' + str(np.round(info_dict['average_mae'], 2)) + '_' + info_dict['model_name']


def get_path(
        info_dict: dict,
        folder: tp.Optional[PosixPath] = RESULTS_PATH,
        info_dict2folder_name: tp.Optional[tp.Callable] = _default_info_dict2_folder_name
        ) -> PosixPath:
    """

    Args:
        info_dict (dict):
        folder (tp.Optional[PosixPath], optional):
        info_dict2folder_name (tp.Optional[tp.Callable], optional):

    Returns:
        PosixPath: path to store logs
    """
    return folder / info_dict['target_name'].replace(':', '_') \
          / info_dict2folder_name(info_dict) \
          / str(hash(json.dumps(info_dict)))
        #   / clean_str(json.dumps(info_dict['model_params']) + '__' + json.dumps(info_dict['model_transforms']))


def save_results(
        ts: etna.datasets.tsdataset.TSDataset,
        pipe: etna.pipeline.Pipeline,
        metrics_df: pd.DataFrame,
        n_folds: tp.Union[int, tp.List[etna.pipeline.FoldMask]],
        forecast_df: pd.DataFrame,
        fold_info_df: pd.DataFrame,
        target_name: str,
        info_dict: tp.Optional[dict] = None,
        folder: PosixPath = RESULTS_PATH,
        comment: tp.Optional[str] = None,
        save_figure: bool = False
) -> None:
    """saves everything into folder

    Args:
        ts (etna.datasets.tsdataset.TSDataset):
        pipe (etna.pipeline.Pipeline):
        metrics_df (pd.DataFrame):
        n_folds (tp.Union[int, tp.List[etna.pipeline.FoldMask]]):
        forecast_df (pd.DataFrame):
        fold_info_df (pd.DataFrame):
        target_name (str):
        info_dict (tp.Optional[dict], optional):
        folder (tp.Optional[PosixPath], optional):
        comment (tp.Optional[str], optional):
        save_figure (bool): whether save
            forecasts vs true figure in html format
        
    """
    info_dict = info_dict or pipeline2dict(ts, pipe, n_folds, forecast_df, target_name, comment)
    print(info_dict)
    folder = get_path(info_dict, folder)
    print(folder)

    try:
        os.makedirs(folder, exist_ok=False)
    except FileExistsError:
        print(
            f'warning: folder {folder} already exists, perhaps, this experiment was already processed',
            file=sys.stderr
        )
        os.makedirs(folder, exist_ok=True)

    with open(folder / 'pipe.pkl', 'wb') as f:
        pickle.dump(pipe, f)

    for name, df in zip(['metrics.pkl', 'forecast.pkl', 'fold_inf.pkl'],
                        [metrics_df, forecast_df, fold_info_df]):
        with open(folder / name, 'wb') as f:
            pickle.dump(df, f)

    with open(folder / 'info.json', 'w') as f:
        json.dump(info_dict, f, indent=1, ensure_ascii=False)

    if save_figure:
        fig = get_figure(ts, forecast_df, target_name)
        fig.write_html(folder / 'last_month_backtest_results.html')
