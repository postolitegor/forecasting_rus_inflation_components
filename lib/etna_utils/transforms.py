import pandas as pd
import warnings

from dateutil.relativedelta import relativedelta
from etna.transforms.base import OneSegmentTransform, ReversiblePerSegmentWrapper
from typing import List, Optional


FILE = 'temp.txt'

# Class for processing one segment.
class _OneSegmentTsIntegrateTransform(OneSegmentTransform):

    # Constructor with the name of the column to which the transformation will be applied.
    def __init__(self, in_column: str, inplace: bool = True, out_column: Optional[str] = None):
        """
        Create instance of _OneSegmentTsIntegrateTransform.

        Parameters
        ----------
        tr_column:
            name of processed column
        """
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = out_column
       

    def _get_column_name(self) -> str:
            if self.inplace:
                return self.in_column
            elif self.out_column:
                return self.out_column
            else:
                return self.__repr__()


    # Provide the necessary training. For example calculates the coefficients of a linear trend.
    # In this case, we calculate the indices that need to be changed
    # and remember the old values for inverse transform.
    def fit(self, df: pd.DataFrame) -> "_OneSegmentFloorCeilTransform":
        """
        Do nothing

        Returns
        -------
        self
        """
        return self

    # Apply changes.
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply time series integrate transform
        and remember results

        Parameters
        ----------
        df:
            DataFrame to transform

        Returns
        -------
        transformed series
        """
        result_df = df
        ser = result_df[self.in_column]
        ser = (1 + ser / 100).cumprod()
        result_df[self.in_column] = ser

        self.transformed_ts = result_df.copy()

        return result_df

    # Returns back changed values.
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transformation for transform. Return back changed values.

        Parameters
        ----------
        df:
            data to transform

        Returns
        -------
        pd.DataFrame
            reconstructed data
        """
        result = df
        ser = result[self.in_column]

        # select last date to add
        first_date = ser.index[0]
        if first_date > self.transformed_ts.index[0]:
            last_ts_date = self.transformed_ts[self.transformed_ts.index < first_date].index[-1]
            last_ts_val =  self.transformed_ts[self.transformed_ts.index < first_date].iloc[-1].values[0]
        else:
            last_ts_date = self.transformed_ts.index[0] + relativedelta(months=-1)
            last_ts_val = 1

        ser.loc[last_ts_date] = last_ts_val
        ser.sort_index(inplace=True)

        ser = ((ser / ser.shift() - 1) * 100)
        ser = ser.iloc[1:]

        result[self.in_column] = ser
        return result


class TsIntegrateTransform(ReversiblePerSegmentWrapper):
    """Transform that truncate values to an interval [ceil, floor]"""

    def __init__(self, in_column: str, inplace: bool = True, out_column: Optional[str] = None):
        """Create instance of FloorCeilTransform.
        Parameters
        ----------
        in_column:
            name of processed column
        """
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = out_column
        super().__init__(
            transform=_OneSegmentTsIntegrateTransform(
            in_column=self.in_column,
            inplace=inplace,
            out_column=out_column
        ),
            required_features=[in_column],
        )

    # Here we need to specify output columns with regressors, if transform creates them.
    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.

        Returns
        -------
        :
            List with regressors created by the transform.
        """
        return []
