
rcp_name_to_color = {
    'RCP45': 'tab:purple',
    'RCP85': 'r',
}

def get_rcp_label(rcp_name: str) -> str:
    return f'{rcp_name[:-1]}.{rcp_name[-1]}'