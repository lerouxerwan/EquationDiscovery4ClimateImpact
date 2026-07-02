import pandas as pd
import xarray as xr


def get_df_for_a_season(da: xr.DataArray, extract_winter: bool) -> pd.DataFrame:
    """ If extract_winter is True, we extract winter, otherwise we extract spring"""
    # Remove the last month of December
    da = da[:-1]
    # Extract the winter mean or spring mean
    if extract_winter:
        winter_data = da.where(da.time.dt.month.isin([12, 1, 2]))
        winter_year = xr.where(
            winter_data.time.dt.month == 12,
            winter_data.time.dt.year + 1,
            winter_data.time.dt.year
        )
        # winter_data = winter_data.assign_coords(winter_year=winter_year)
        means = winter_data.groupby(winter_year).mean()
    else:
        spring_data = da.where(da.time.dt.month.isin([3, 4, 5]))
        means = spring_data.groupby("time.year").mean()[1:]
    return means.to_dataframe()
