

ssp_name_to_color = {
    'SSP370': 'gold',
    'SSP585': 'darkred',
}

def get_ssp_label(ssp_name: str) -> str:
    return f'{ssp_name[:-2]} {ssp_name[-2]}.{ssp_name[-1]}'