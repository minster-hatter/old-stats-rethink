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
from arviz import (
    from_pymc3,
    summary,
    plot_ppc,
    plot_trace,
    plot_posterior,
    plot_pair,
)
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
    mu = Normal("mu", 178.0, 20.0)
    sigma = Uniform("sigma", 0.0, 50.0)
    # Likelihood.
    h_i = Normal("h_i", mu, sigma, observed=adults_data["height"])
    # Predictive checks and sampling.
    prior_pc_m_4_1 = sample_prior_predictive()
    trace_m_4_1 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_4_1 = sample_posterior_predictive(trace_m_4_1)
    idata_m_4_1 = from_pymc3(
        trace_m_4_1,
        prior=prior_pc_m_4_1,
        posterior_predictive=post_pc_m_4_1,
        model=m_4_1,
    )

# Summary and plots for model m_4_1.
model_to_graphviz(m_4_1).render("m_4_1_dag", cleanup=True, format="png")

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
savefig("m_4_1_posterior_mu_sigma.png")

plot_pair(idata_m_4_1, var_names=["mu", "sigma"], kind="kde")
savefig("m_4_1_pairplot_mu_sigma.png")

# New model with the same structure and an overly narow prior on mu.
with Model() as m_4_2:
    """h_i ~ Normal(mu, sigma)
    mu ~ Normal(178, 0.1)
    sigma ~ Uniform(0, 50)
    """
    # Priors.
    mu = Normal("mu", 178.0, 0.1)
    sigma = Uniform("sigma", 0.0, 50.0)
    # Likelihood.
    h_i = Normal("h_i", mu, sigma, observed=adults_data["height"])
    # Prior predictive checks and sampling.
    prior_pc_m_4_2 = sample_prior_predictive()
    trace_m_4_2 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_4_2 = sample_posterior_predictive(trace_m_4_2)
    idata_m_4_2 = from_pymc3(
        trace_m_4_2,
        prior=prior_pc_m_4_2,
        posterior_predictive=post_pc_m_4_2,
        model=m_4_2,
    )

# Summary and plots for m_4_2.
model_to_graphviz(m_4_2).render("m_4_2_dag", cleanup=True, format="png")

summary(idata_m_4_2, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_4_2_summary.csv"
)

plot_ppc(
    idata_m_4_2, num_pp_samples=PREDICTIVE_SAMPLES, mean=False, group="prior"
)
savefig("m_4_2_prior_pc.png")

plot_ppc(idata_m_4_2, num_pp_samples=PREDICTIVE_SAMPLES, mean=False)
savefig("m_4_2_posterior_pc.png")

plot_trace(idata_m_4_2, compact=True)
savefig("m_4_2_traces.png")

plot_posterior(
    idata_m_4_2, var_names=["mu", "sigma"], hdi_prob=CI, point_estimate="median"
)
savefig("m_4_2_posterior_mu_sigma.png")
