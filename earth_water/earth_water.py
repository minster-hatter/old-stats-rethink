from pymc3 import (
    Model,
    Uniform,
    Binomial,
    sample_prior_predictive,
    sample,
    sample_posterior_predictive,
)
from arviz import from_pymc3, summary, plot_ppc
from matplotlib.pyplot import savefig
from numpy import median

# Constants.
SAMPLES = int(1e3)
CHAINS = 5
N = 9
W_OBSERVED = 6

# Binomial model for water vs. land being observed.
with Model() as m:
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
        trace, prior=prior_pc, posterior_predictive=post_pc, model=m
    )

# Output summary and plots.
summary(idata, hdi_prob=0.9, stat_funcs=[median]).to_csv("m_summary.csv")
plot_ppc(idata, mean=False, group="prior")
savefig("m_prior.png")
plot_ppc(idata, mean=False)
savefig("m_posterior.png")
