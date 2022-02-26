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
    hdi,
    plot_hdi,
)
from numpy import median, arange
from numpy.random import randint
from matplotlib.pyplot import savefig, subplots, scatter, xlabel, ylabel

# Constants to be used later.
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

# Create feature for later use.
W = adults_data["weight"] - adults_data["weight"].mean()

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

plot_pair(idata_m_4_2, var_names=["mu", "sigma"], kind="kde")
savefig("m_4_2_pairplot_mu_sigma.png")


# New model with weight as a linear predictor.
with Model() as m_4_3_trial:
    """h_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta * (x_i - x_bar)
    alpha ~ Normal(178, 20)
    beta ~ Normal(0, 10)
    sigma ~ Uniform(0, 50)
    """
    # Priors.
    alpha = Normal("alpha", 178.0, 20.0)
    beta = Normal("beta", 0.0, 10.0)
    sigma = Uniform("sigma", 0.0, 50.0)
    # Likelihood.
    mu_i = alpha + beta * W
    h_i = Normal("h_i", mu_i, sigma, observed=adults_data["height"])
    # Prior predictive check.
    prior_pc_m_4_3_trial = sample_prior_predictive()
    idata_m_4_3_trial = from_pymc3(
        prior=prior_pc_m_4_3_trial, model=m_4_3_trial
    )

# Check if the prior assumptions make sense.
plot_ppc(
    idata_m_4_3_trial,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    group="prior",
)
savefig("m_4_3_trial_prior_pc.png")

# A new model is created with more sensible prior h_i predictions.
with Model() as m_4_3:
    """h_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta * (x_i - x_bar)
    alpha ~ Normal(178, 20)
    beta ~ Normal(0, 1)  # Note that this is now 1, not 10.
    sigma ~ Uniform(0, 50)
    """
    # Priors.
    alpha = Normal("alpha", 178.0, 20.0)
    beta = Normal("beta", 0.0, 1.0)
    sigma = Uniform("sigma", 0.0, 50.0)
    # Likelihood.
    mu = alpha + beta * W
    h_i = Normal("h_i", mu, sigma, observed=adults_data["height"])
    # Prior predictive check.
    prior_pc_m_4_3 = sample_prior_predictive()
    idata_m_4_3_trial = from_pymc3(prior=prior_pc_m_4_3, model=m_4_3)

# Find that the new prior assumptions are more sensible.
plot_ppc(
    idata_m_4_3_trial,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    group="prior",
)
savefig("m_4_3_prior_pc.png")

# Fit the model and explore the outputs.
with m_4_3:
    # Sampling and predictive checks.
    trace_m_4_3 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_4_3 = sample_posterior_predictive(trace_m_4_3)
    idata_m_4_3 = from_pymc3(
        trace_m_4_3,
        prior=prior_pc_m_4_3,
        posterior_predictive=post_pc_m_4_3,
        model=m_4_3,
    )

model_to_graphviz(m_4_3).render("m_4_3_dag", cleanup=True, format="png")

summary(idata_m_4_3, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_4_3_summary.csv"
)

plot_ppc(
    idata_m_4_3, num_pp_samples=PREDICTIVE_SAMPLES, mean=False, group="prior"
)
savefig("m_4_3_prior_pc.png")

plot_ppc(idata_m_4_3, num_pp_samples=PREDICTIVE_SAMPLES, mean=False)
savefig("m_4_3_posterior_pc.png")

plot_trace(idata_m_4_3, compact=True)
savefig("m_4_3_traces.png")

plot_posterior(idata_m_4_3, hdi_prob=CI, point_estimate="median")
savefig("m_4_3_posteriors.png")

plot_pair(idata_m_4_3, kind="kde")
savefig("m_4_3_pairplots.png")

# Plot credible models for the average relationship.
random_indices = randint(len(trace_m_4_3), size=20)
fig, ax = subplots(1, 1)
ax.scatter(
    adults_data["weight"], adults_data["height"], color="black", marker="x"
)
for index in random_indices:
    ax.plot(
        adults_data["weight"],
        trace_m_4_3["alpha"][index]
        + trace_m_4_3["beta"][index]
        * (adults_data["weight"] - adults_data["weight"].mean()),
        "orangered",
        alpha=0.2,
    )
ax.set_xlabel("weight (kg)")
ax.set_ylabel("height (cm)")
savefig("m_4_3_model_plot.png")
