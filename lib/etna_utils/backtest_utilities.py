import contextlib
import datetime
import functools
import itertools
import io
import numpy as np
import os
import pandas as pd
import traceback
import typing as tp

from pathlib import Path, PosixPath

import mpire

from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE, MSE, SMAPE
from etna.pipeline import Pipeline, FoldMask

import etna_utils.log_results


START_TEST_DATE = pd.to_datetime('2019-01-01')
RESULTS_PATH = Path().absolute() / 'results'


DATE_TYPE = tp.Union[pd._libs.tslibs.timestamps.Timestamp, str]


def get_nfolds_int(
        first_test_timestamp: DATE_TYPE,
        ts
) -> int:
    """calculates number of folds for etna backtest, such that forecast_df
    starts from first_tets_timestamp and ends at last date of ts

    Args:
        first_test_timestamp (DATE_TYPE): _description_
        ts (_type_): _description_

    Returns:
        int: n_folds for etna backtest
    """
    last_test_timestamp = max(ts.to_pandas().index)
 
    return int(np.round(
        (last_test_timestamp - first_test_timestamp) / np.timedelta64(1, 'M')
    )) + 1


def run_backtest(
        ts: TSDataset,
        pipe: Pipeline,
        n_folds: tp.Union[int, tp.List[FoldMask]],
        metrics: tp.Optional[tp.List] = None,
        stride: int = 1,
        pipe_stdout_file: tp.Optional[PosixPath] = None,
        pipe_stderr_file: tp.Optional[PosixPath] = None
) -> tp.Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """run backtest and try to redirect prints
       (redirection does not work for Prophet outputs)

    Args:
        ts (TSDataset):
        pipe (Pipeline):
        n_folds (tp.Union[int, tp.List[FoldMask]]):
        metrics (tp.Optional[tp.List], optional):
        stride (int, optional):
        pipe_stdout_file (tp.Optional[PosixPath], optional):
        pipe_stderr_file (tp.Optional[PosixPath], optional):

    Returns:
        tp.Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    pipe_stdout_file = pipe_stdout_file or os.devnull
    pipe_stderr_file = pipe_stderr_file or os.devnull

    metrics = metrics or [MAE()]

    with create_and_open(pipe_stdout_file, 'a') as stdout,\
         create_and_open(pipe_stderr_file, 'a') as stderr:
        with contextlib.redirect_stdout(stdout),\
             contextlib.redirect_stderr(stderr):
            metrics_df, forecast_df, fold_info_df = pipe.backtest(
                ts=ts,
                metrics=metrics,
                n_folds = n_folds,
                stride=stride,
                joblib_params=dict(verbose=0)
            )
    
    return pipe, metrics_df, forecast_df, fold_info_df


def get_true_horizon_forecast(
        forecast_df: pd.DataFrame,
        col_name: tp.Optional[str] = None
) -> pd.DataFrame:
    """Get forecasts only for the last period.
    Given horizon > 1, etna returns all forecasts
    (for one period ahead, two periods, etc).
    This function selects forecasts only up to 1 period

    Args:
        forecast_df (pd.DataFrame):
        col_name (tp.Optional[str], optional):

    Returns:
        pf.DataFrame: forecast_df with the last forecast only
    """
    forecast_df = forecast_df[col_name] 
    forecast_df = forecast_df.groupby('fold_number').tail(1)

    return forecast_df


# =====================================================
# run and save function
# =====================================================

def run_and_save(
        ts: TSDataset,
        pipe,
        start_test_date: DATE_TYPE,
        target_name: str,
        pipe_stdout_file: tp.Optional[PosixPath] = None,
        pipe_stderr_file: tp.Optional[PosixPath] = None,
        folder: PosixPath = RESULTS_PATH,
        comment: str = '',
        save_results: tp.Callable = etna_utils.log_results.save_results
) -> None:
    """run pipe and save results via save_results function

    Args:
        ts (TSDataset):
        pipe (_type_):
        start_test_date (DATE_TYPE):
        target_name (str):
        pipe_stdout_file (tp.Optional[PosixPath], optional):
        pipe_stderr_file (tp.Optional[PosixPath], optional):
        folder (tp.Optional[PosixPath], optional):
        comment (str, optional):
        bsave_results (tp.Callable, optional):
    """
    n_folds = get_nfolds_int(start_test_date, ts)
    p, metrics_df, forecast_df, fold_info_df = run_backtest(
        ts,
        pipe,
        n_folds,
        pipe_stdout_file=pipe_stdout_file,
        pipe_stderr_file=pipe_stderr_file
    )

    save_results(
        ts,
        p,
        metrics_df,
        n_folds,
        forecast_df,
        fold_info_df,
        target_name=target_name,
        folder=folder,
        comment=comment
    )


# =========================================================
# wite logs utilities
# =========================================================


@contextlib.contextmanager
def create_and_open(
    file_name: str,
    option: str
) -> io.TextIOWrapper:
    """contextmanager that create folders if they are note exist
    And not throwing error like simple "open"

    Args:
        file_name (str):
        option (str): 'a', 'w', 'wb' like options

    Returns:
        io.TextIOWrapper:

    Yields:
        io.TextIOWrapper: 
    """
    try:
        with open(file_name, option) as f:
            yield f
    except FileNotFoundError:
        os.makedirs(os.path.join(*os.path.split(file_name)[:-1]))
        with open(file_name, option) as f:
            yield f


def write_exception(
        file_path: PosixPath,
        exceptions = None #  tp.Optional[tp.Union[Exception, tp.Iterable[Exception]]]
):
    exceptions = exceptions or Exception
    def decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except exceptions:
                with create_and_open(file_path, 'a') as f:
                    print(str(datetime.datetime.now()), file=f)
                    print('args:', args, file=f)
                    print('kwargs:', kwargs, file=f)
                    traceback.print_exc(file=f)
                    print('=' * 80, file=f)

        return _wrapper
    return decorator


# ==========================================================
# multiprocessing
# ==========================================================

def prepare_shared_objects(
    start_test_date = START_TEST_DATE,
    stdout_path: PosixPath = Path('Logs') / 'stdout.txt',
    stderr_path: PosixPath = Path('Logs') / 'stderr.txt',
    folder: PosixPath = RESULTS_PATH
) -> tp.Tuple:
    """

    Args:
        start_test_date (_type_, optional): date to compute test metrics from
        stdout_path (PosixPath, optional): file to redirect stdout
            of pipe (does not work for Prophet models)
        stderr_path (PosixPath, optional): file to redirect stderr
            of pipe (does not work for Prophet models)
        folder (tp.Optional[PosixPath], optional): folder to store all results

    Returns:
        tp.Tuple: currently stores tuple of objects
    """
    shared_objects = (
        start_test_date,
        stdout_path,
        stderr_path,
        folder
    )
    return shared_objects


@write_exception(Path('logs') / 'failed_pipes_errors.txt')
def run_pipe(
    shared_objects: tp.Tuple,
    ts: TSDataset,
    pipe: Pipeline
) -> None:
    """wrapper function for multiprocessing

    Args:
        shared_objects (tp.Tuple):
        ts (TSDataset):
        pipe (Pipeline):
    """
    start_test_date, pipe_stdout_file, pipe_stderr_file, folder = shared_objects
    target_name = ts.segments[0] #  TODO: rewrite code for multisegment

    run_and_save(
        ts,
        pipe,
        start_test_date,
        target_name,
        pipe_stdout_file,
        pipe_stderr_file,
        folder
    )


def run_pipes_cartesian_product_mp(
        ts_datasets: tp.Iterable[TSDataset],
        pipes: tp.Iterable[Pipeline],
        n_jobs: tp.Optional[int] = None,
        *args,
        start_test_date=START_TEST_DATE,
        stdout_path: PosixPath = Path('logs') / 'stdout.txt',
        stderr_path: PosixPath = Path('logs') / 'stderr.txt',
        folder: PosixPath = RESULTS_PATH
):
    """implements:
        for ts in ts_datasets:
            for pipe in pipes:
                run pipe
    Args:
        ts_datasets (tp.Iterable[TSDataset]): 
        pipes (tp.Iterable[Pipeline]): 
    """
    shared_objects = prepare_shared_objects(
        start_test_date,
        stdout_path,
        stderr_path,
        folder
    )
    stdout_path: PosixPath = Path('logs') / 'stdout.txt'
    stderr_path: PosixPath = Path('logs') / 'stderr.txt'

    with create_and_open(stdout_path, 'a') as stdout,\
         create_and_open(stdout_path, 'a') as stderr:
      with contextlib.redirect_stdout(stdout),\
             contextlib.redirect_stderr(stderr):
        with mpire.WorkerPool(
        n_jobs=n_jobs,
        shared_objects=shared_objects,
        daemon=False
        ) as pool:
            for _ in pool.imap_unordered(
                run_pipe,
                itertools.product(ts_datasets, pipes),
                progress_bar=True,
                iterable_len=len(ts_datasets) * len(pipes)
            ):
                pass


def run_pipes_mp(
        tss_pipes: tp.List,
        # ts_datasets: tp.Iterable[TSDataset],
        # pipes: tp.Iterable[Pipeline],
        n_jobs: tp.Optional[int] = None,
        *args,
        start_test_date=START_TEST_DATE,
        stdout_path: PosixPath = Path('logs') / 'stdout.txt',
        stderr_path: PosixPath = Path('logs') / 'stderr.txt',
        folder: PosixPath = RESULTS_PATH
):
    """implements:
        for ts in ts_datasets:
            for pipe in pipes:
                run pipe
    Args:
        tss_pipes (tp.List[(ts, pipe)])
    """
    shared_objects = prepare_shared_objects(
        start_test_date,
        stdout_path,
        stderr_path,
        folder
    )
    stdout_path: PosixPath = Path('logs') / 'stdout.txt'
    stderr_path: PosixPath = Path('logs') / 'stderr.txt'

    with create_and_open(stdout_path, 'a') as stdout,\
         create_and_open(stdout_path, 'a') as stderr:
      with contextlib.redirect_stdout(stdout),\
             contextlib.redirect_stderr(stderr):
        with mpire.WorkerPool(
        n_jobs=n_jobs,
        shared_objects=shared_objects,
        daemon=False
        ) as pool:
            for _ in pool.imap_unordered(
                run_pipe,
                # itertools.product(ts_datasets, pipes),
                tss_pipes,
                progress_bar=True,
                # iterable_len=
            ):
                pass



#  deprecated, but maybe useful later
'''
def collect_loo_fold_masks(
        horizon: int,
        train_window_size: tp.Optional[int] = None,
        first_train_timestamp: tp.Optional[DATE_TYPE] = None,
        first_test_time_stamp: tp.Optional[DATE_TYPE] = None,
        last_test_time_stamp: tp.Optional[DATE_TYPE] = None,
        freq='MS'
        #    last_train_timestamp: tp.Optional[DATE_TYPE] = None,
):
    """_summary_

    Args:
        horizon (int): _description_
        train_window_size (tp.Optional[int], optional): _description_. Defaults to None.
        first_train_timestamp (tp.Optional[DATE_TYPE], optional): _description_. Defaults to None.
        first_test_time_stamp (tp.Optional[DATE_TYPE], optional): _description_. Defaults to None.
        last_test_time_stamp (tp.Optional[DATE_TYPE], optional): _description_. Defaults to None.
        freq (str, optional): _description_. Defaults to 'MS'.
    """
    first_test_time_stamp = first_test_time_stamp or pd.to_datetime('2021-01-01')
    last_test_time_stamp = last_test_time_stamp or pd.to_datetime('2023-04-01')
    first_train_timestamp = first_train_timestamp or pd.to_adtetime('2003-01-01')

    masks = []
    for test_timestamp in pd.date_range(
        start=first_test_time_stamp,
        end=last_test_time_stamp,
        freq=freq
    ):
        train_st = 
        train_end = 
        target_timestamp = 
        mask = FoldMask(
            first_train_timestamp = train_st,
            last_train_timestamp = train_end,
            target_
        )
'''


# def make_pipelines(
#         model,
#         transforms: tp.List,
#         horizons=[1, 2, 3, 6, 12, 24, 36]
# ) -> tp.List[Pipeline]:
#     pipelines = []
#     horizon_info = []

#     for horizon in horizons:
#         pipelines.append(
#             Pipeline(
#                 model=model,
#                 transforms=transforms,
#                 horizon=horizon
#             )
#         )
#         horizon_info.append({'horizon': horizon})
#     return pipelines, horizon_info
