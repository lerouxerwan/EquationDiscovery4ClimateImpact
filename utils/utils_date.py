

def get_month_names() -> list[str]:
    """Return a copy of the list month_names"""
    return ['January', 'February', 'March', 'April', 'May', 'June', 'July',
            'August', 'September', 'October', 'November', 'December']


def get_short_month_names() -> list[str]:
    return [name[:3] for name in get_month_names()]

def get_season_short_names():
    return ['DJF', 'MAM', 'JJA', 'SON']


month_to_name = dict(enumerate(get_month_names()))
month_to_short_name = dict(enumerate(get_short_month_names()))
