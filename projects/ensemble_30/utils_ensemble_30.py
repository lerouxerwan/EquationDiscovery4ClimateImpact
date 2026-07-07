from pathlib import Path

from utils.utils_path import DATA_PATH

WEBPATH = "https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/DATA_NEMOMED12"

ENSEMBLE_30_PATH = Path(DATA_PATH) / '30_members_ensemble'
RAW_PATH = ENSEMBLE_30_PATH / "raw"
