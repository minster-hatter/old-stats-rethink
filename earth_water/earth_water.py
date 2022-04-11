from configparser import ConfigParser

from pymc3 import (
    Model,
    Uniform,
    Binomial,
    sample_prior_predictive,
    sample,
    sample_posterior_predictive,
    model_to_graphviz,
)
from arviz import from_pymc3, summary, plot_ppc, plot_trace, plot_posterior
from matplotlib.pyplot import savefig
from numpy import median

# Constants.
config = ConfigParser()
config.read("../config.ini")
SAMPLES = config.getint("parameters", "SAMPLES")
CHAINS = config.getint("parameters", "CHAINS")
PREDICTIVE_SAMPLES = config.getint("parameters", "PREDICTIVE_SAMPLES")
CI = config.getfloat("parameters", "CREDIBLE_INTERVAL")
N = 9
W_OBSERVED = 6

# Binomial model for water vs. land being observed.
with Model() as m_2_6:
    """W ~ Binomial(N, p)
    p ~ Uniform(0, 1)
    """
    # Priors.
    p = Uniform("p", 0.0, 1.0)
    # Likelihood.
    W = Binomial("W", n=N, p=p, observed=W_OBSERVED)
    # Sampling and output packaging.
    prior_pc = sample_prior_predictive()
    trace = sample(draws=SAMPLES, chains=CHAINS)
    post_pc = sample_posterior_predictive(trace)
    idata = from_pymc3(
        trace, prior=prior_pc, posterior_predictive=post_pc, model=m_2_6
    )

# Output summary and plots.
summary(idata, hdi_prob=CI, stat_funcs=[median]).to_csv("m_2_6_summary.csv")

plot_ppc(idata, num_pp_samples=PREDICTIVE_SAMPLES, mean=False, group="prior")
savefig("m_2_6_prior_pc.png")

plot_ppc(idata, num_pp_samples=PREDICTIVE_SAMPLES, mean=False)
savefig("m_2_6_posterior_pc.png")

plot_trace(idata, compact=True)
savefig("m_2_6_traces.png")

plot_posterior(idata, var_names=["p"], hdi_prob=CI, point_estimate="median")
savefig("m_2_6_posterior_p.png")

model_to_graphviz(m_2_6).render("m_2_6_dag", cleanup=True, format="png")
