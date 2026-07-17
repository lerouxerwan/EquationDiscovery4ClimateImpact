import pandas as pd
import xarray as xr
from xarray import DataArray


def get_df_for_a_season(da_month_time_series: xr.DataArray, extract_winter: bool, variable_name: str) -> pd.DataFrame:
    """ If extract_winter is True, we extract winter, otherwise we extract spring"""
    # Extract the winter mean or spring mean
    means = get_da_seasonal_mean(da_month_time_series, extract_winter)
    # Create df and rename index and columns
    df = means.to_dataframe()
    df.index.name = 'year'
    df.rename(columns={variable_name: f'{variable_name}_{extract_winter}'}, inplace=True)
    return df


def get_da_seasonal_mean(da_month_time_series: DataArray, extract_winter: bool) -> DataArray:
    if extract_winter:
        winter_data = da_month_time_series.where(da_month_time_series.time.dt.month.isin([12, 1, 2]))
        winter_year = xr.where(
            winter_data.time.dt.month == 12,
            winter_data.time.dt.year + 1,
            winter_data.time.dt.year
        )
        # winter_data = winter_data.assign_coords(winter_year=winter_year)
        means = winter_data.groupby(winter_year).mean()
    else:
        spring_data = da_month_time_series.where(da_month_time_series.time.dt.month.isin([3, 4, 5]))
        means = spring_data.groupby("time.year").mean()
    return means
