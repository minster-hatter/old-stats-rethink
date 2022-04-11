from configparser import ConfigParser
from pathlib import Path
from sqlite3 import connect

from pandas import read_sql

# Constants to be used later.
config = ConfigParser()
config.read("../config.ini")
SAMPLES = config.getint("parameters", "SAMPLES")
CHAINS = config.getint("parameters", "CHAINS")
PREDICTIVE_SAMPLES = config.getint("parameters", "PREDICTIVE_SAMPLES")
CI = config.getfloat("parameters", "CREDIBLE_INTERVAL")

# Add database location to the path and import.
data_path = Path(__file__).parent / "../data/stats_rethink.db"
cnxn = connect(data_path)
milk_data = read_sql("SELECT * FROM milk", cnxn)
print(milk_data.describe())
