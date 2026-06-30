import xarray as xr


def get_df_from_two_years_file(variable: str, extract_winter: bool, weights: xr.DataArray, sorted_variable_files: list[str]):
    """ If extract_winter is True, we extract winter, otherwise we extract spring"""
    # Loop to extract variable for the area of interest
    time_series_list = []
    for f in sorted_variable_files:
        # second_year = variable_file_to_second_year[f]
        ds = xr.open_dataset(f)
        da = ds[variable].weighted(weights)
        time_series = da.mean(dim='x').mean(dim='y')
        time_series_list.append(time_series.copy())
    # Combine time series together
    da = xr.concat(time_series_list, dim='time')
    da = da[11:-1]
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
    df = means.to_dataframe()
    df.rename(columns={variable: f'{variable}_{extract_winter}'}, inplace=True)
    df.index.name = 'year'
    return df
