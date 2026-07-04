

def get_filename(variable_name: str, ensemble_id: int) -> str:
    return f'{"{:03d}".format(ensemble_id)}ENS04_1d_19790627_20201227_{variable_name}.nc'
