import typing as tp

from etna.models import (
    AutoARIMAModel,
    BATSModel,
    ElasticPerSegmentModel,
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

Results = tp.Tuple[tp.List[dict], tp.Type] #  return list of kwargs and model class

def get_naive_models(
    lags = [1, 2, 12]
) -> Results:
    kw_args = [{'lag': l} for l in lags]
    return kw_args, NaiveModel


def get_arima_models() -> Results:
    return [], AutoARIMAModel


def get_bats_models() -> Results:
    return [], BATSModel


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
    for g in ['linear', 'logistic']:
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
    args = []
    return args, TBATSModel
    