import pandas as pd
import typing as tp

import etna


def df2ts_univariate(
        df: pd.DataFrame,
        col_name: str
) -> etna.datasets.tsdataset.TSDataset:
    """make DataFrame -> TSDataset transform


    Args:
        df (pd.DataFrame): CPIMOM DataFrame
        col_name (str): column to put to 

    Returns:
        etna.datasets.tsdataset.TSDataset: 
            TSDataset, containing one segment only
    """
    df = df[col_name].reset_index()    
    df.columns = ['timestamp', 'target']
    df['segment'] = col_name

    ts = etna.datasets.tsdataset.TSDataset.to_dataset(df)
    ts = etna.datasets.tsdataset.TSDataset(ts, freq='MS')

    return ts


def prepare_univariate_datasets(
        mom_df: pd.DataFrame,
        columns: tp.Optional[str] = None
) -> tp.List[etna.datasets.tsdataset.TSDataset]:
    """df2ts_univariate for any column in mom_df

    Args:
        mom_df (pd.DataFrame): 
        columns (tp.Optional[str], optional):

    Returns:
        tp.List[etna.datasets.tsdataset.TSDataset]: 
    """
    if columns is None:
        columns = mom_df.columns

    tss = []
    for col in columns:
        ts = df2ts_univariate(mom_df, col)
        tss.append(ts)
    
    return tss


def collect_exog_df(
        dfs_all: tp.Dict[str, tp.Dict[str, pd.DataFrame]],
        other_names: tp.Iterable[tp.Tuple],
        segment_name: str,
        new_names_dict: tp.Optional[dict] = None,
        exog_df_shift: int = 2
    ) -> pd.DataFrame:
    """prepares dataframe for exogenous TSDataset

    Args:
        dfs_all (tp.Dict[str, tp.Dict[str, pd.DataFrame]]): other dataaframes (format as from load_others func from load_data_utils)
        other_names (tp.Iterable[tp.Tuple]): names to store to result dataset, example:
            [
                ('ProducerPriceInflation', 'ppmmo2', 'PPI: OKVED2: Prev Month=100: Mfg: Textiles (f11)')
            ]
        segment_name (str): example: "Fruits&Vegetables"
        new_names_dict (tp.Optional[dict], optional): new names in final dataframe

    Returns:
        pd.DataFrame:
    """
    
    tdfs = []
    for names in other_names:
        tdfs.append(
            dfs_all[names[0]][names[1]][names[2]]
        )
    exog_df = pd.concat(tdfs, axis=1)
    exog_df.reset_index(inplace=True)
    exog_df.rename(columns={'date': 'timestamp'}, inplace=True)
    
    if new_names_dict is not None:
        exog_df.rename(columns=new_names_dict, inplace=True)
    exog_df['segment'] = segment_name

    exog_df = exog_df.loc[:, ~exog_df.columns.duplicated()]
    
    exog_df = exog_df.shift(exog_df_shift).iloc[exog_df_shift:, :]

    return exog_df


def prepare_multivariate_dataset(
    target_ser: pd.Series,
    dfs_all: tp.Dict[str, tp.Dict[str, pd.DataFrame]],
    other_names: tp.Iterable[tp.Tuple],
    new_names_dict: tp.Optional[dict] = None,
    exog_df_shift = 2
) -> etna.datasets.tsdataset.TSDataset:
    df = target_ser.to_frame().reset_index()
    df.columns = ['timestamp', 'target']
    df['segment'] = target_ser.name
    target_ts = etna.datasets.tsdataset.TSDataset.to_dataset(df)

    df = collect_exog_df(
        dfs_all,
        other_names,
        target_ser.name,
        new_names_dict,
        exog_df_shift=exog_df_shift
    )
    exog_ts = etna.datasets.tsdataset.TSDataset.to_dataset(df)

    full_ts = etna.datasets.tsdataset.TSDataset(
        target_ts,
        freq='MS',
        df_exog=exog_ts,
        known_future=[]
    )

    return full_ts


def tsdataset_dropna(
        ts: etna.datasets.tsdataset.TSDataset
    ) -> etna.datasets.tsdataset.TSDataset:
    """
    drop nas from ts
    Args:
        ts (etna.datasets.tsdataset.TSDataset):

    Returns:
        etna.datasets.tsdataset.TSDataset: ts without na
    """
    mask = ts.to_pandas().isna().any(axis=1)
    first_date = ts.to_pandas()[~mask].index.min()
    last_date = ts.to_pandas()[~mask].index.max()
    return etna.datasets.tsdataset.TSDataset(
        ts.raw_df[first_date: last_date]
        ,df_exog=ts.df_exog[first_date: last_date]
        ,freq='MS'
        ,known_future=[]
    )


def tsdataset_dropna2(
    ts: etna.datasets.tsdataset.TSDataset
) -> etna.datasets.tsdataset.TSDataset:
    """
    drop nas from ts
    Args:
        ts (etna.datasets.tsdataset.TSDataset):

    Returns:
        etna.datasets.tsdataset.TSDataset:
    """
    mask = ts.to_pandas().isna().any(axis=1)
    first_date = ts.to_pandas()[~mask].index.min()
    last_date = ts.to_pandas()[~mask].index.max()
    return etna.datasets.tsdataset.TSDataset(
        ts[first_date: last_date]
        ,freq='MS'
        ,known_future=[]
    )
