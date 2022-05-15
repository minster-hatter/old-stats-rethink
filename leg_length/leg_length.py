from configparser import ConfigParser

from numpy import vstack, median
from numpy.random import normal, uniform
from pandas import DataFrame
from pymc3 import (
    Model,
    Normal,
    Exponential,
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
    plot_forest,
)
from matplotlib.pyplot import savefig

# Constants to be used later.
config = ConfigParser()
config.read("../config.ini")
SAMPLES = config.getint("parameters", "SAMPLES")
CHAINS = config.getint("parameters", "CHAINS")
PREDICTIVE_SAMPLES = config.getint("parameters", "PREDICTIVE_SAMPLES")
CI = config.getfloat("parameters", "CREDIBLE_INTERVAL")

# Create the data.
N = 100  # The number of individuals.
height = normal(10, 2, N)
leg_proportion = uniform(0.4, 0.5, N)
# Create leg lengths with Gaussian measurement error.
leg_length = leg_proportion * height
left_leg_length = leg_length + normal(0, 0.2, N)
right_leg_length = leg_length + normal(0, 0.2, N)

length_data = DataFrame(
    vstack([height, left_leg_length, right_leg_length]).T,
    columns=["height", "left_leg_length", "right_leg_length"],
)
print(length_data.head())

# Linear model relating height to the lengths of both legs.
with Model() as m_6_1:
    # Priors.
    alpha = Normal("alpha", 10.0, 100.0)
    beta_left = Normal("beta_left", 2, 10)
    beta_right = Normal("beta_right", 2, 10)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu = (
        alpha
        + beta_left * length_data["left_leg_length"]
        + beta_right * length_data["right_leg_length"]
    )
    height = Normal("height", mu, sigma, observed=length_data["height"])
    # Sample and extract.
    prior_pc_m_6_1 = sample_prior_predictive()
    trace_m_6_1 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_6_1 = sample_posterior_predictive(trace_m_6_1)
    idata_m_6_1 = from_pymc3(
        trace_m_6_1,
        prior=prior_pc_m_6_1,
        posterior_predictive=post_pc_m_6_1,
        model=m_6_1,
    )

model_to_graphviz(m_6_1).render("m_6_1_dag", cleanup=True, format="png")

summary(idata_m_6_1, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_6_1_summary.csv"
)

plot_ppc(
    idata_m_6_1,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
)
savefig("m_6_1_posterior_pc.png")

plot_trace(
    idata_m_6_1,
    compact=True,
    var_names=["alpha", "beta_left", "beta_right", "sigma"],
)
savefig("m_6_1_traces.png")

plot_posterior(
    idata_m_6_1,
    hdi_prob=CI,
    var_names=["alpha", "beta_left", "beta_right", "sigma"],
    kind="hist",
    color="orangered",
    point_estimate="median",
)
savefig("m_6_1_posterior_hisograms")

plot_pair(
    idata_m_6_1,
    var_names=["alpha", "beta_left", "beta_right", "sigma"],
    kind="kde",
)
savefig("m_6_1_pairplot_alpha_beta_N_sigma.png")

plot_forest(
    idata_m_6_1,
    var_names=["alpha", "beta_left", "beta_right"],
    combined=True,
    hdi_prob=CI,
    colors="black",
)
savefig("m_6_1_forest_plot.png")

# A linear model using only the left leg length.
