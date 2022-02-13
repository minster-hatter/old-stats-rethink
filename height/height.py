from pathlib import Path
from sqlite3 import connect

from pandas import read_sql

# Add database location to the path and import.
data_path = Path(__file__).parent / "../data/stats_rethink.db"
cnxn = connect(data_path)
howell_data = read_sql("SELECT * FROM howell1", cnxn)

# Summarise and explore the data.
print(howell_data.describe())
