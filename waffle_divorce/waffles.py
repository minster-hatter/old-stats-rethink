from pathlib import Path
from sqlite3 import connect

from pandas import read_sql
from scikitlearn.preprocessing import scale

# Constants to be used later.
SAMPLES = int(1e3)
CHAINS = 5
PREDICTIVE_SAMPLES = int(1e2)
CI = 0.9
FS = 14

# Add database location to the path and import.
data_path = Path(__file__).parent / "../data/stats_rethink.db"
cnxn = connect(data_path)
waffles_data = read_sql("SELECT * FROM waffles", cnxn)
print(waffles_data.describe())

# Standardize the divorce and median age at marriage fields.
waffles_data["D"] = scale(waffles_data["Divorce"])
waffles_data["A"] = scale(waffles_data["MedianAgeMarriage"])
