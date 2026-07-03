from pathlib import Path

import xarray as xr

from utils.utils_path import DATA_PATH



# def compute_weights() -> xr.DataArray:
#     MASK_PATH = Path(DATA_PATH) / "mask"
#     # Compute all weights
#     mesh_mask_MED = xr.open_dataset(MASK_PATH / "mesh_mask_MED12_v3.6_75lev.nc")
#     lat = mesh_mask_MED['e2t'][0, :, :]
#     lon = mesh_mask_MED['e1t'][0, :, :]
#     weights = lat * lon
#     # Keep only LION4 weights
#     gol4_mask = xr.open_dataset(MASK_PATH / "subbasins_dev_MED12.nc")['LION4']
#     gol4_mask = gol4_mask.fillna(False)
#     weights = weights.where(gol4_mask, drop=False)
#     weights = weights.fillna(0)
#     return weights

def get_df_rcsm6(model: str, scenario: str):
    variable_name_and_extract_winter = [('tos', True), ('tos', False), ('sos', False), ('rsntds', True)]
    df_list = [get_df(model, scenario, variable_name, extract_winter)
               for (variable_name, extract_winter) in variable_name_and_extract_winter]
    df = pd.concat(df_list, axis=1)
    # Convert to the correct unit
    for i, column_name in enumerate(df.columns):
        if column_name.startswith('tos'):
            df.iloc[:, i] += 273.15
    # Rename columns
    df.rename(columns={column_name: get_new_column_name(column_name) for column_name in df.columns}, inplace=True)
    # Add NPP columns
    df['NPP'] = 106 + 6.2 * df['SSS_MAM'] - 0.086 * df['SST_DJF'] - 1.03 * df['SST_MAM'] + 0.11 * df['Shortwave_DJF']
    return df

def get_df(model: str, scenario: str) -> pd.DataFrame:
    filepath = Path(DATA_PATH) / model / scenario / "cache.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, index_col=0)
    else:
        df = _get_df(model, scenario)
        df.to_csv(filepath)
    return df

def main(area: str):
    """For each mask we extract the four features, compute NPP, and save the Dataframe"""
    pass

if __name__ == '__main__':
    for area in ['LION4']:
        main(area)
