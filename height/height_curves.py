# Polynomials linking height and weight that are later compared.
from pathlib import Path
from sqlite3 import connect

# Constants to be used later.
SAMPLES = int(1e3)
CHAINS = 5
PREDICTIVE_SAMPLES = int(1e2)
CI = 0.9

# Add database location to the path and import.
data_path = Path(__file__).parent / "../data/stats_rethink.db"
cnxn = connect(data_path)
howell_data = read_sql("SELECT * FROM howell1", cnxn)

