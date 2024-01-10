import typing as tp

from etna.models import (
    AutoARIMAModel,
    BATSModel,
    HoltModel,
    HoltWintersModel,
    MovingAverageModel,
    NaiveModel,
    ProphetModel,
    SARIMAXModel,
    SeasonalMovingAverageModel,
    SimpleExpSmoothingModel,
    TBATSModel
)

from ..pipeline_utilities import get_pipelines

Results = tp.Tuple[tp.List[dict], tp.Type] #  return list of kwargs and model class

def get_naive_models(
    lags = [1, 2, 12]
) -> Results:
    kw_args = [{'lag': l} for l in lags]
    return kw_args, NaiveModel


def get_arima_models() -> Results:
    return [], AutoARIMAModel


def get_bats_models() -> Results:
    return [{'n_jobs': 1}], BATSModel


# def get_elastic_per_segment_models() -> Results:
#     alphas = [0.01, 0.2, 0.3, 0.5, 0.7, 1, 2, 4]
#     l1_ratios = [0.1, 0.3, 0.6, 0.9, 1]

#     res = []
#     for a in alphas:
#         for l1 in l1_ratios:
#             kw_args = {
#                 'alpha': a,
#                 'l1_ratio': l1,
#             }
#             res.append(kw_args)

#     return res, ElasticPerSegmentModel


def get_holt_models() -> Results:
    return [], HoltModel


def get_holtwinters_models() -> Results:
    args = []
    for dtrend in [True, False]:
        for seasonal_period in [None, 12]:
            kw_args = {
                'damped_trend': dtrend,
                'seasonal_periods': seasonal_period
            }
        args.append(kw_args)
    return args, HoltWintersModel


def get_ma_models() -> Results:
    args = []
    for w in [1, 2, 3, 5]:
        args.append({'window': w})

    return args, MovingAverageModel


def get_prophet_models() -> Results:
    args = []
    # for g in ['linear', 'logistic']:  # has to set capacity
    for g in ['linear']:
        for cr in [0.1, 0.5, 0.9]: #  [0.1, 0.3, 0.5, 0.8, 0.9]:
            for ys in [True]: #  [True, False]:
                for sp in [0.1, 0.5, 0.9]: #  [0.1, 0.3, 0.5, 0.8, 0.9]:
                    for cp in [0.1, 0.5, 0.9]: #  [0.1, 0.3, 0.5, 0.8, 0.9]:
                        args.append({
                            'growth': g,
                            'changepoint_range': cr,
                            'yearly_seasonality': ys,
                            'seasonality_prior_scale': sp,
                            'changepoint_prior_scale': cp
                        })
    return args, ProphetModel


def get_seasonal_ma_models() -> Results:
    args = []
    for window in [1, 2, 3]:
        for seasonality in [3, 4, 6, 12]:
            args.append({
                'window': window,
                'seasonality': seasonality
            })
    return args, SeasonalMovingAverageModel


def get_simle_exp_models() -> Results:
    args = []
    return args, SimpleExpSmoothingModel


def get_tbats_models() -> Results:
    args = [{'n_jobs':1}]
    return args, TBATSModel


def get_sarimax_models() -> Results:
    """docs are here
    https://etna-docs.netlify.app/api/etna.models.sarimax.sarimaxmodel

    Returns:
        Results: _description_
    """
    args = []
    for p in range(3):
        for d in range(3):
            for q in range(3):
                for P in range(2):
                    for Q in range(2):
                        for D in range(2):
                            for s in [4, 12]:
                                # for trend in [[]]:
                                    args.append({
                                        'order': (p, d, q),
                                        'seasonal_order': (P, D, Q, s)
                                    })

    return args, SARIMAXModel
    

def load_univariate_pipelines(
        horizons=[1, 2, 3, 6, 12, 24, 36],
        transforms=[],
):
    def _get_local_function_names():
        res = []
        for k, v in globals().copy().items():
            if callable(v) \
                and v.__module__ == __name__ \
                and k != '_get_local_function_names':
                res.append(v)
        return res
    
    pipes = []
    for func in _get_local_function_names():
        if func.__name__.startswith('get'):
            args, model_class = func()
            pipes.extend(
                get_pipelines(
                    horizons=horizons,
                    transforms=transforms,
                    kw_args=args,
                    ModelClass=model_class,
                    pipe=None
                )
            )
    return pipes
