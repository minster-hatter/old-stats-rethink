from pathlib import Path
from sqlite3 import connect

from pandas import read_sql
from pymc3 import (
    Model,
    Normal,
    Uniform,
    sample_prior_predictive,
    sample,
    sample_posterior_predictive,
    model_to_graphviz,
)
from arviz import from_pymc3, summary, plot_ppc, plot_trace, plot_posterior
from numpy import median
from matplotlib.pyplot import savefig

# Constants used later.
SAMPLES = int(1e3)
CHAINS = 5
PREDICTIVE_SAMPLES = int(1e2)
CI = 0.9

# Add database location to the path and import.
data_path = Path(__file__).parent / "../data/stats_rethink.db"
cnxn = connect(data_path)
howell_data = read_sql("SELECT * FROM howell1", cnxn)
adults_data = howell_data[howell_data["age"] >= 18]

# Summarise and explore the data.
print("Summary of the Howell data for adults:\n", adults_data.describe(), "\n")

# Linear model of adult height.
with Model() as m_4_1:
    """h_i ~ Normal(mu, sigma)
    mu ~ Normal(178, 20)
    sigma ~ Uniform(0, 50)
    """
    # Priors.
    mu = Normal("mu", 178, 20)
    sigma = Uniform("sigma", 0, 50)
    # Likelihood.
    h_i = Normal("h_i", mu, sigma, observed=adults_data["height"])
    # Predictive checks and sampling.
    prior_pc = sample_prior_predictive()
    trace = sample(SAMPLES, chains=CHAINS)
    post_pc = sample_posterior_predictive(trace)
    idata_m_4_1 = from_pymc3(
        trace, prior=prior_pc, posterior_predictive=post_pc, model=m_4_1
    )

# Summary and plots for model m_4_1.
summary(idata_m_4_1, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_4_1_summary.csv"
)

plot_ppc(
    idata_m_4_1, num_pp_samples=PREDICTIVE_SAMPLES, mean=False, group="prior"
)
savefig("m_4_1_prior_pc.png")

plot_ppc(idata_m_4_1, num_pp_samples=PREDICTIVE_SAMPLES, mean=False)
savefig("m_4_1_posterior_pc.png")

plot_trace(idata_m_4_1, compact=True)
savefig("m_4_1_traces.png")

plot_posterior(
    idata_m_4_1, var_names=["mu", "sigma"], hdi_prob=CI, point_estimate="median"
)
savefig("m_4_1_posterior_mu_sigma")

model_to_graphviz(m_4_1).render("m_4_1_dag", cleanup=True, format="png")
