from pathlib import Path
from sqlite3 import connect

from numpy import linspace, quantile, asarray, median, percentile
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
    model_to_graphviz,
)
from pymc3.math import dot
from arviz import from_pymc3, summary, plot_ppc, plot_trace, plot_posterior

# Constants to be used later.
SAMPLES = int(1e3)
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

model_to_graphviz(m_4_7).render("m_4_7_dag", cleanup=True, format="png")

summary(idata_m_4_7, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_4_7_summary.csv"
)

plot_ppc(
    idata_m_4_7, num_pp_samples=PREDICTIVE_SAMPLES, mean=False, group="prior"
)
savefig("m_4_7_prior_pc.png")

plot_ppc(idata_m_4_7, num_pp_samples=PREDICTIVE_SAMPLES, mean=False)
savefig("m_4_7_posterior_pc.png")

plot_trace(idata_m_4_7, compact=True, var_names=["alpha", "w", "sigma"])
savefig("m_4_7_traces.png")

plot_posterior(
    idata_m_4_7,
    hdi_prob=CI,
    var_names=["alpha", "w", "sigma"],
    kind="hist",
    color=CHERRY_BLOSSOM_PINK,
)
savefig("m_4_7_posterior_hisograms")

fig, ax = subplots(3, 1)
for i in range(N_KNOTS + 2):
    ax[0].plot(blossom_data["year"], (B[:, i]), color="black")
ax[0].set_ylabel("Basis", fontsize=FS)
wp = trace_m_4_7[w].mean(0)
for i in range(N_KNOTS + 2):
    ax[1].plot(blossom_data["year"], (wp[i] * B[:, i]), color="black")
ax[1].set_xlim(812, 2015)
ax[1].set_ylim(-6, 6)
ax[1].set_xlabel("Year", fontsize=FS)
ax[1].set_ylabel("Basis", fontsize=FS)
low, high = percentile(post_pc_m_4_7["D"], [10, 90], axis=0)
ax[2].plot(blossom_data["year"], blossom_data["doy"], "x", color="black")
ax[2].fill_between(
    blossom_data["year"], low, high, alpha=0.25, color=CHERRY_BLOSSOM_PINK
)
ax[2].plot(
    blossom_data["year"], post_pc_m_4_7["D"].mean(axis=0), color=CHERRY_BLOSSOM_PINK
)
ax[2].set_xlabel("Year", fontsize=FS)
ax[2].set_ylabel("Day of Year", fontsize=FS)
