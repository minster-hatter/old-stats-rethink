from pathlib import Path
from sqlite3 import connect

from pandas import read_sql
from pymc3 import Model, Normal, Uniform, sample_prior_predictive, sample, sample_posterior_predictive
froma arviz import from_pymc3, summary, plot_ppc

# Add database location to the path and import.
data_path = Path(__file__).parent / "../data/stats_rethink.db"
cnxn = connect(data_path)
howell_data = read_sql("SELECT * FROM howell1", cnxn)
adults_data = howell_data[howell_data["age"] >= 18]

# Summarise and explore the data.
print("Summary of the Howell data for adults:\n", adults_data.describe(), "\n")

with Model() as m_4_1:
    """h_i ~ Normal(mu, sigma)
    mu ~ Normal(178, 20)
    sigma ~ Uniform(0, 50)
    """
    None
