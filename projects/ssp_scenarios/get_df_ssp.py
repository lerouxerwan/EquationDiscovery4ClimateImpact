from pathlib import Path

import pandas as pd
import xarray as xr

from utils.utils_path import DATA_PATH

RCSM6B_PATH = Path(DATA_PATH) / "RCSM6B"



def compute_weights():
    # Compute all weights
    mesh_mask_MED = xr.open_dataset(RCSM6B_PATH / "mesh_mask_MED12_v3.6_75lev.nc")
    lat = mesh_mask_MED['e2t'][0, :, :]
    lon = mesh_mask_MED['e1t'][0, :, :]
    weights = lat * lon
    # Keep only LION4 weights
    gol4_mask = xr.open_dataset(RCSM6B_PATH / "subbasins_dev_MED12.nc")['LION4']
    gol4_mask = gol4_mask.fillna(False)
    weights = weights.where(gol4_mask, drop=False)
    weights = weights.fillna(0)
    return weights


def _get_df(scenario: str, weights: xr.DataArray, variable: str, extract_winter: bool):
    """ If extract_winter is True, we extract winter, otherwise we extract spring"""
    folder_path = RCSM6B_PATH / scenario
    assert folder_path.exists()
    variable_files = [f for f in folder_path.iterdir() if f.name.startswith(variable)]
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

def get_df_rcsm6b(scenario: str):
    couples = [
        ('tos', True),
        ('tos', False),
        ('sos', False),
        ('rsntds', True),
    ]
    weights = compute_weights()
    df = pd.concat([_get_df(scenario, weights, v, b) for (v, b) in couples], axis=1)
    df['NPP'] = 106 + 6.2 * df['SSS_MAM'] - 0.086 * df['SST_DJF'] - 1.03 * df['SST_MAM'] + 0.11 * df['Shortwave_DJF']
    return df


if __name__ == '__main__':
    for s in ['SSP585', 'SSP370'][:]:
        df = get_df_rcsm6b(s)
        print(df.head())
