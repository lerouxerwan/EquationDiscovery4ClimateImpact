import xarray as xr


def get_df_from_two_years_file(variable: str, extract_winter: bool, weights: xr.DataArray, variable_files: list[str]):
    """ If extract_winter is True, we extract winter, otherwise we extract spring"""
    variable_file_to_second_year = {f: int(str(f)[-9:-5]) for f in variable_files}
    sorted_variable_files = sorted(variable_files, key=lambda x: variable_file_to_second_year[x])
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
        season_str = 'DJF'
        winter_data = da.where(da.time.dt.month.isin([12, 1, 2]))
        winter_year = xr.where(
            winter_data.time.dt.month == 12,
            winter_data.time.dt.year + 1,
            winter_data.time.dt.year
        )
        # winter_data = winter_data.assign_coords(winter_year=winter_year)
        means = winter_data.groupby(winter_year).mean()
    else:
        season_str = 'MAM'
        spring_data = da.where(da.time.dt.month.isin([3, 4, 5]))
        means = spring_data.groupby("time.year").mean()[1:]
    # Convert to the correct unit
    if variable == 'tos':
        means += 273.15
    # Rename the column
    variable_to_name = {
        'tos': "SST",
        'sos': "SSS",
        'rsntds': "Shortwave",
    }
    df = means.to_dataframe()
    df.rename(columns={variable: f'{variable_to_name[variable]}_{season_str}'}, inplace=True)
    df.index.name = 'year'
    return df
