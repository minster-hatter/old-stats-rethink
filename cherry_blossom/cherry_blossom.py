from pathlib import Path
from sqlite3 import connect

from numpy import linspace, quantile, asarray
from pandas import read_sql
from patsy import dmatrix
from matplotlib.pyplot import subplots, savefig
from pymc3 import (
    Model,
    Normal,
    Exponential,
    Deterministic,
    sample_prior_predictive,
    sample,
    sample_posterior_predictive,
)
from pymc3.math import dot
from arviz import from_pymc3, summary

# Constants to be used later.
SAMPLES = int(1e2)
CHAINS = 5
PREDICTIVE_SAMPLES = int(1e2)
CI = 0.9
FS = 14
CHERRY_BLOSSOM_PINK = (1, 183 / 255, 197 / 255)

# Add database location to the path and import.
data_path = Path(__file__).parent / "../data/stats_rethink.db"
cnxn = connect(data_path)
blossom_data = read_sql("SELECT * FROM kyoto", cnxn)
print(blossom_data.describe())

# Create the design matrix.
N_KNOTS = 15
KNOT_SPACING = linspace(0, 1, N_KNOTS)
KNOT_POSITIONS = quantile(blossom_data["year"], KNOT_SPACING)
B = dmatrix(
    "bs(year, knots=knots, degree=3, include_intercept=True) - 1",
    {"year": blossom_data["year"], "knots": KNOT_POSITIONS[1:-1]},
)

# Plot the blossom date data.
fig, ax = subplots()
ax.plot(blossom_data["year"], blossom_data["doy"], color=CHERRY_BLOSSOM_PINK)
ax.set_xlabel("Year", fontsize=FS)
ax.set_ylabel("Blossom Emergence Day", fontsize=14)
savefig("cherry_blossom_day_of_year.png")

# Fit the penalized splines model.
with Model() as m_4_7:
    """D ~ Normal(mu_i, sigma)
    mu_i = alpha + SUM_{k=1}^{K}(w_k * B_{k, i})
    alpha ~ Normal(100, 10)
    w_j ~ Normal(0, 10)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha = Normal("alpha", 100.0, 10.0)
    w = Normal("w", 0.0, 1.0, shape=B.shape[1])
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu = Deterministic("mu", alpha + dot(asarray(B, order="F"), w.T))
    D = Normal("D", mu, sigma, observed=blossom_data["doy"])
    # Sampling and predictive checks.
    prior_pc_m_4_7 = sample_prior_predictive()
    trace_m_4_7 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_4_7 = sample_posterior_predictive(trace_m_4_7)
    idata_m_4_7 = from_pymc3(
        trace_m_4_7,
        prior=prior_pc_m_4_7,
        posterior_predictive=post_pc_m_4_7,
        model=m_4_7,
    )
