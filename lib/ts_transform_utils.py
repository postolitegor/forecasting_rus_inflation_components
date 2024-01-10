import typing as tp
from copy import copy

import numpy as np
import pandas as pd


def ts_integrate(
    df: tp.Union[pd.DataFrame, pd.Series],
    fill_na=False,
    fill_first_one=True,
    first_ind: tp.Optional[pd.Timestamp] = None,
) -> tp.Union[pd.DataFrame, pd.Series]:
    """Integrates time series.

    Args:
        df (tp.Union[pd.DataFrame, pd.Series]): data to integrate (it should be like '1%, 2%, 0.1%' format)
        fill_na (bool, optional): whether to fill na with values, or left them na like
        fill_first_one (bool, optional): whether to fill first value with ones or not
        first_ind (tp.Optional[pd.Timestamp], optional): first index (that is filled with 1), if None than the change between the first 2 indices is used

    Returns:
        tp.Union[pd.DataFrame, pd.Series]: _description_
    """
    df = copy(df)

    return_df = True
    if isinstance(df, pd.Series):
        df = df.to_frame()
        return_df = False

    if fill_first_one:
        if first_ind is None:
            # assert df.index[1] - df.index[0] == df.index[2] - df.index[1]
            first_ind = df.index[0] - (df.index[1] - df.index[0])
        first = pd.DataFrame(
            np.zeros(df.shape[1]), index=[first_ind], columns=df.columns
        )

        df = pd.concat([first, df])

    if not fill_na:
        prev_na = df.isna()

    int_vals = np.empty((df.shape[0], df.shape[1]))
    mul_val = np.ones(df.shape[1])
    for i, row in enumerate(df.itertuples(index=False)):
        mul_val *= 1 + np.nan_to_num(row[0:]) / 100
        int_vals[i] = mul_val.copy()

    df[df.columns] = int_vals

    if not fill_na:
        df = df.mask(prev_na)
    if return_df:
        return df
    else:
        return df.iloc[:, 0]


def ts_differentiate(
    df: tp.Union[pd.DataFrame, pd.Series], fill_na=False, drop_first: bool = True
) -> pd.DataFrame:
    """reverse to ts_integrate.

    Args:
        df (tp.Union[pd.DataFrame, pd.Series]): data to differentiate
        fill_na (bool, optional):

    Returns:
        pd.DataFrame:
    """
    return_df = True
    if isinstance(df, pd.Series):
        df = df.to_frame()
        return_df = False
    if not fill_na:
        prev_na = df.isna()
    df = copy(df)

    np.seterr(divide="ignore", invalid="ignore")
    if not fill_na:
        prev_na = df.isna()

    df = df.fillna(method="ffill")

    df = copy(df)
    prev_vals = np.ones(df.shape[1])
    vals = np.empty((df.shape[0], df.shape[1]))

    for i, row in enumerate(df.itertuples(index=False)):
        # vals[i] = (np.nan_to_num(row[0:]) - prev_vals) / prev_vals * 100
        vals[i] = (np.nan_to_num(row) / prev_vals - 1) * 100
        prev_vals = np.nan_to_num(row[0:])

    df[df.columns] = vals
    if not fill_na:
        df = df.mask(prev_na)

    if drop_first:
        df = df.iloc[1:]

    if return_df:
        return df
    else:
        return df.iloc[:, 0]


# def test_integrate():
# on weekly inflation
#     assert (ts_integrate(ts_differentiate(int_ser.dropna())) == int_ser.dropna()).all()
#     assert (ts_differentiate(ts_integrate(w_ser.iloc[226:])) == w_ser).all()
