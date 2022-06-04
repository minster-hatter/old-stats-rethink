from configparser import ConfigParser

from numpy import repeat, median
from numpy.random import normal, binomial
from pandas import DataFrame
from arviz import summary, from_pymc3, plot_ppc, plot_trace, plot_posterior
from pymc3 import (
    Model,
    Lognormal,
    Exponential,
    Normal,
    sample_prior_predictive,
    sample,
    sample_posterior_predictive,
    model_to_graphviz,
    Deterministic,
)
from matplotlib.pyplot import savefig

# Constants to be used later.
config = ConfigParser()
config.read("../config.ini")
SAMPLES = config.getint("parameters", "SAMPLES")
CHAINS = config.getint("parameters", "CHAINS")
PREDICTIVE_SAMPLES = config.getint("parameters", "PREDICTIVE_SAMPLES")
CI = config.getfloat("parameters", "CREDIBLE_INTERVAL")

N = 100
h_0 = normal(10, 2, N)
treatment = repeat([0, 1], N / 2)
fungus = binomial(n=1, p=0.5 - treatment * 0.4, size=N)
h_1 = h_0 + normal(5 - 3 * fungus, size=N)

plant_data = DataFrame.from_dict(
    {"h_0": h_0, "h_1": h_1, "treatment": treatment, "fungus": fungus}
)
print(plant_data.head())

# Make a model with p as the growth proportion between timepoints.
with Model() as m_6_6:
    """h_{1, i} ~ Normal(mu_i, sigma)
    mu_i = h_{0, i} * p
    p ~ Log-Normal(0, 0.25)
    sigma ~ Exponential(1)
    """
    # Priors.
    p = Lognormal("p", 0.0, 0.25)
    sigma = Exponential("sigma", 1.0)
    # Liklelihood.
    mu = p * plant_data["h_0"]
    h_1 = Normal("h_1", mu, sigma, observed=plant_data["h_1"])
    # Inference and extract.
    prior_pc_m_6_6 = sample_prior_predictive()
    trace_m_6_6 = sample(SAMPLES, chains=CHAINS)
    posterior_pc_m_6_6 = sample_posterior_predictive(trace_m_6_6)
    idata_m_6_6 = from_pymc3(
        trace_m_6_6,
        prior=prior_pc_m_6_6,
        posterior_predictive=posterior_pc_m_6_6,
        model=m_6_6,
    )

M_6_6_VARIABLES = ["p", "sigma"]

model_to_graphviz(m_6_6).render("m_6_6_dag", cleanup=True, format="png")

summary(idata_m_6_6, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_6_6_summary.csv"
)

plot_ppc(
    idata_m_6_6,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
    group="prior",
)
savefig("m_6_6_prior_pc.png")

plot_ppc(
    idata_m_6_6,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
)
savefig("m_6_6_posterior_pc.png")

plot_trace(idata_m_6_6, compact=True, var_names=M_6_6_VARIABLES)
savefig("m_6_6_traces.png")

plot_posterior(
    idata_m_6_6,
    hdi_prob=CI,
    var_names=M_6_6_VARIABLES,
    kind="hist",
    color="orangered",
    point_estimate="median",
)
savefig("m_6_6_posterior_hisograms")

# Make a model with p as a function of treatment and fungus status.
with Model() as m_6_7:
    """h_{1, i} ~ Normal(mu_i, sigma)
    mu_i = h_{0, i} * p
    p = alpha + beta_T * T_i + beta_F * F_i
    alpha ~ Log-Normal(0, 0.25)
    beta_T ~ Normal(0, 0.5)
    beta_F ~ Normal(0, 0.5)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha = Lognormal("alpha", 0.0, 0.2)
    beta_T = Normal("beta_T", 0.0, 0.5)
    beta_F = Normal("beta_F", 0.0, 0.5)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    p = alpha + beta_T * plant_data["treatment"] + beta_F * plant_data["fungus"]
    mu = p * plant_data["h_0"]
    h_1 = Normal("h_1", mu, sigma, observed=plant_data["h_1"])

    # Inference and extract.
    prior_pc_m_6_7 = sample_prior_predictive()
    trace_m_6_7 = sample(SAMPLES, chains=CHAINS)
    posterior_pc_m_6_7 = sample_posterior_predictive(trace_m_6_7)
    idata_m_6_7 = from_pymc3(
        trace_m_6_7,
        prior=prior_pc_m_6_7,
        posterior_predictive=posterior_pc_m_6_7,
        model=m_6_7,
    )

M_6_7_VARIABLES = ["beta_T", "beta_F", "sigma"]

model_to_graphviz(m_6_7).render("m_6_7_dag", cleanup=True, format="png")

summary(idata_m_6_7, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_6_7_summary.csv"
)

plot_ppc(
    idata_m_6_7,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
    group="prior",
)
savefig("m_6_7_prior_pc.png")

plot_ppc(
    idata_m_6_7,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
)
savefig("m_6_7_posterior_pc.png")

plot_trace(idata_m_6_7, compact=True, var_names=M_6_7_VARIABLES)
savefig("m_6_7_traces.png")

plot_posterior(
    idata_m_6_7,
    hdi_prob=CI,
    var_names=M_6_7_VARIABLES,
    kind="hist",
    color="orangered",
    point_estimate="median",
)
savefig("m_6_7_posterior_hisograms")
