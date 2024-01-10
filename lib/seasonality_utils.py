import platform
import typing as tp

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import sys
import traceback

try:
    from fbprophet import Prophet
except ModuleNotFoundError:
    from prophet import Prophet

from pathlib import Path

from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tools.sm_exceptions import X13Error


_IS_WINDOWS = platform.system() == "Windows"


def deseason(
    ser, period=12, model="additive", two_sided=False, return_decomp=False
) -> tp.Union[pd.Series, DecomposeResult]:
    """simple seasonal decomposition.

    :param str model: 'additive' or 'multiplicative'
    """
    decomp = sm.tsa.seasonal_decompose(
        ser, model=model, period=period, two_sided=two_sided
    )
    if return_decomp:
        return decomp
    ans = ser - decomp.seasonal
    ans.name = ser.name
    return ans


def deseason_stl(
    ser,
    period=12,
    seasonal_deg=1,
    robust=True,
    return_decomp=False,
    *stl_args,
    **stl_kwargs,
):
    """STL using LOESS decomposition."""
    decomp = STL(
        ser,
        period=period,
        seasonal_deg=seasonal_deg,
        robust=robust,
        *stl_args,
        **stl_kwargs,
    ).fit()
    if return_decomp:
        return decomp

    ans = ser - decomp.seasonal
    ans.name = ser.name
    return ans


def detrend_stl(
    ser,
    period=12,
    seasonal_deg=1,
    robust=True,
    return_decomp=False,
    *stl_args,
    **stl_kwargs,
):
    decomp = STL(
        ser,
        period=period,
        seasonal_deg=seasonal_deg,
        robust=robust,
        *stl_args,
        **stl_kwargs,
    ).fit()
    if return_decomp:
        return decomp
    ans = ser - decomp.trend
    ans.name = ser.name
    return ans


def deseason_prophet(ser: pd.Series, return_fcst=False, *pr_args, **pr_kwargs):
    """prophet deseason."""
    model = Prophet(*pr_args, **pr_kwargs)
    x = ser.to_frame().reset_index()
    x.columns = ["ds", "y"]
    model.fit(x)

    dates_df = model.make_future_dataframe(periods=0)
    fcst = model.predict(dates_df)
    fcst.set_index("ds", inplace=True)
    if model.seasonality_mode == "additive":
        ser_des = ser - fcst["additive_terms"]
    else:
        ser_des = ser - fcst["multiplicative_terms"] * fcst["trend"]

    if return_fcst:
        return fcst
    return ser_des


def deseason_x13(ser: pd.Series,
                 return_result=False,
                 max_const_fightbug=10,
                 print_error=False,
                 *x13_args,
                 **x13_kwargs):
    xpath = Path(__file__).parent / "sarimax13utils"
    if _IS_WINDOWS:
        xpath /= "windows_files"
    else:
        xpath /= "unix_files"
    xpath = str(xpath)

    def _run_x13(ser, log=None):
        try:
            res = sm.tsa.x13.x13_arima_analysis(
                endog=ser,
                x12path=xpath,
                prefer_x13=True,
                # outlier=True,
                log=log,
                *x13_args,
                **x13_kwargs,
            )
        except AttributeError:
                res = sm.tsa.x13_arima_analysis(
                    endog=ser,
                    x12path=xpath,
                    prefer_x13=True,
                    # outlier=True,
                    log=log,
                    *x13_args,
                    **x13_kwargs,
                )
        return res
    
    def _run_all_logs(ser):
        for log_bool in [None, True, False]:
            try:
                res = _run_x13(ser, log=log_bool)
                return res
            except X13Error:
                pass
        raise X13Error
    
    # this is to fight bug in arima13, no better idea was found
    for const in range(max_const_fightbug):
        try:
            ser_p = ser + const
            res = _run_all_logs(ser_p)
            break
        except X13Error:
            if print_error:
                print(traceback.format_exc())
                print(f'tried adding all constants between 0 and {max_const_fightbug}')
            
            if const == max_const_fightbug - 1:
                raise


    if return_result:
        return res

    ser = pd.Series(res.seasadj.values, index=ser.index)

    return ser


def plot_seasonal_decompose(
    result: DecomposeResult,
    dates: pd.Series = None,
    title: str = "Seasonal Decomposition",
):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name="Observed"),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name="Trend"),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name="Seasonal"),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="lines", name="Residual"),
            row=4,
            col=1,
        )
        .update_layout(
            height=900,
            title=f"<b>{title}</b>",
            margin={"t": 100},
            title_x=0.5,
            showlegend=False,
        )
    )
