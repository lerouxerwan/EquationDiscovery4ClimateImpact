from typing import Optional

import pandas as pd


def load_units(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Optional[list[str]], pd.Series, Optional[list[str]]]:
    """Extract the row 'UNIT' then remove it from df (if the row exists)"""
    if 'UNIT' in df.index:
        series_units = df.loc['UNIT']
        X_units = _load_units(series_units.iloc[1:])
        y_units = _load_units(series_units.iloc[:1])
        df = df.iloc[1:, :]
        # See https://symbolicml.org/DynamicQuantities.jl/dev/units/ for a list of accepted units
    else:
        X_units, y_units = None, None
    series_y = df.iloc[:, 0].astype(float)
    df_X = df.iloc[:, 1:].astype(float)
    return df, df_X, X_units, series_y, y_units

def _load_units(series_units: pd.Series) -> list[str]:
    return [unit if isinstance(unit, str) else '' for unit in series_units.to_list()]