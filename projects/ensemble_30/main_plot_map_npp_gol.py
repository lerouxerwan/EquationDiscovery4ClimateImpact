import xarray as xr
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_split.utlis_plot_selected_equation import get_label
from projects.ensemble_30.df_for_ensemble_30 import get_da_seasonal
from projects.scenarios.utils_get_df_rcsm6 import compute_weights
from utils.utils_plot import show_or_save_plot


def _get_da_seasonal(variable_name: str, ensemble_id: int):
    # Transform variable name
    variable_to_name = {
        'sosstsst': "SST",
        'sosaline': "SSS",
        'qsr': "Shortwave",
    }
    name_to_variable = {v:k for k, v in variable_to_name.items()}
    extract_winter_to_season_name = {
        'True': 'DJF',
        'False': 'MAM',
    }
    season_name_to_extract_winter = {v:k for k, v in extract_winter_to_season_name.items()}
    name, season_name = variable_name.split('_')
    extract_winter = bool(season_name_to_extract_winter[season_name])
    variable = name_to_variable[name]
    return get_da_seasonal(variable, ensemble_id, extract_winter)

def get_da_map_mean(variable_name: str, ensemble_id: int):
    if variable_name == 'NPP':
        da_seasonal = 0.11 * _get_da_seasonal('Shortwave_DJF', ensemble_id)
        da_seasonal +=  106
        da_seasonal += 6.2 * _get_da_seasonal('SSS_MAM', ensemble_id)
        da_seasonal -= 0.086 * _get_da_seasonal('SST_DJF', ensemble_id)
        da_seasonal -= 1.03 * _get_da_seasonal('SST_MAM', ensemble_id)
    else:
        da_seasonal = _get_da_seasonal(variable_name, ensemble_id)
    return da_seasonal.mean(dim='month')

def get_ylabel(variable_name: str):
    if variable_name == 'NPP':
        # Choose long name
        dataset = get_dataset(validation_split=ValidationSplit.NONE)
        y_label = get_label(dataset.target_label)
        return y_label.replace(' (', 'averaged on all\nyears and ensemble members (')
    else:
        return variable_name


def main_plot_map_gol(variable_name: str, ensemble_ids: list[int], show: bool):
    # Averaged over all ensemble members
    da_list = [get_da_map_mean(variable_name, ensemble_id) for ensemble_id in ensemble_ids]
    da_map = xr.concat(da_list, dim='ensemble').mean('ensemble')
    # Zoom on the GOL
    weights = compute_weights(mask_name='MEDNW')
    da_map = da_map.where(weights > 0)
    # Show map
    ax = plt.gca()
    da_map.plot(ax=ax,     cbar_kwargs = {'label': get_ylabel(variable_name)})
    plot_name = f'{variable_name}'
    show_or_save_plot(plot_name, show=show)


if __name__ == '__main__':
    # ensemble_ids =  [1]
    ensemble_ids =  list(range(1, 31))
    for name in ['NPP', 'SSS_MAM', 'SST_MAM', 'SST_DJF', 'Shortwave_DJF'][:]:
        main_plot_map_gol(name, ensemble_ids, show=False)