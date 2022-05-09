from configparser import ConfigParser
from pathlib import Path
from sqlite3 import connect

from pandas import read_sql
from sklearn.preprocessing import scale
from numpy import nan, log10, median
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
from arviz import (
    from_pymc3,
    plot_ppc,
    summary,
    plot_trace,
    plot_posterior,
    plot_pair,
    plot_forest,
)
from matplotlib.pyplot import savefig
from pgmpy.base import DAG

# Constants to be used later.
config = ConfigParser()
config.read("../config.ini")
SAMPLES = config.getint("parameters", "SAMPLES")
CHAINS = config.getint("parameters", "CHAINS")
PREDICTIVE_SAMPLES = config.getint("parameters", "PREDICTIVE_SAMPLES")
CI = config.getfloat("parameters", "CREDIBLE_INTERVAL")

# Add database location to the path and import.
data_path = Path(__file__).parent / "../data/stats_rethink.db"
cnxn = connect(data_path)
milk_data = read_sql("SELECT * FROM milk", cnxn).replace("NaN", nan)

# Standardize the milk data.
milk_data["K"] = scale(milk_data["kcal.per.g"])
milk_data["N"] = scale(milk_data["neocortex.perc"])
log_mass = log10(milk_data["mass"])
milk_data["M"] = scale(log_mass)
print(milk_data.describe())
# Remove NAs to allow m_5_5 to run.
clean_milk_data = milk_data.dropna().copy()
print(clean_milk_data.describe())

# Linear model with milk energy as a function of brain size.
with Model() as m_5_5:
    """K_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_N * N_i
    alpha ~ Normal(0, 0.2)
    beta_N ~ Normal(0, 0.5)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha = Normal("alpha", 0.0, 0.2)
    beta_N = Normal("beta_N", 0.0, 0.5)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu_i = Deterministic("mu_i", alpha + beta_N * clean_milk_data["N"])
    K_i = Normal("K_i", mu_i, sigma, observed=clean_milk_data["K"])
    # Sampling and extracting inference data.
    prior_pc_m_5_5 = sample_prior_predictive()
    trace_m_5_5 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_5_5 = sample_posterior_predictive(trace_m_5_5)
    idata_m_5_5 = from_pymc3(
        trace_m_5_5,
        prior=prior_pc_m_5_5,
        posterior_predictive=post_pc_m_5_5,
        model=m_5_5,
    )
model_to_graphviz(m_5_5).render("m_5_5_dag", cleanup=True, format="png")

summary(idata_m_5_5, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_5_5_summary.csv"
)

plot_ppc(
    idata_m_5_5,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
)
savefig("m_5_5_posterior_pc.png")

plot_trace(idata_m_5_5, compact=True, var_names=["alpha", "beta_N", "sigma"])
savefig("m_5_5_traces.png")

plot_posterior(
    idata_m_5_5,
    hdi_prob=CI,
    var_names=["alpha", "beta_N", "sigma"],
    kind="hist",
    color="orangered",
    point_estimate="median",
)
savefig("m_5_5_posterior_hisograms")

plot_pair(idata_m_5_5, var_names=["alpha", "beta_N", "sigma"], kind="kde")
savefig("m_5_5_pairplot_alpha_beta_N_sigma.png")

# Linear model with milk energy as a function of body mass.
with Model() as m_5_6:
    """K_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_M * M_i
    alpha ~ Normal(0, 0.2)
    beta_M ~ Normal(0, 0.5)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha = Normal("alpha", 0.0, 0.2)
    beta_M = Normal("beta_M", 0.0, 0.5)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu_i = Deterministic("mu_i", alpha + beta_M * clean_milk_data["M"])
    K_i = Normal("K_i", mu_i, sigma, observed=clean_milk_data["K"])
    # Sampling and extracting inference data.
    prior_pc_m_5_6 = sample_prior_predictive()
    trace_m_5_6 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_5_6 = sample_posterior_predictive(trace_m_5_6)
    idata_m_5_6 = from_pymc3(
        trace_m_5_6,
        prior=prior_pc_m_5_6,
        posterior_predictive=post_pc_m_5_6,
        model=m_5_6,
    )

model_to_graphviz(m_5_6).render("m_5_6_dag", cleanup=True, format="png")

summary(idata_m_5_6, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_5_6_summary.csv"
)

plot_ppc(
    idata_m_5_6,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
)
savefig("m_5_6_posterior_pc.png")

plot_trace(idata_m_5_6, compact=True, var_names=["alpha", "beta_M", "sigma"])
savefig("m_5_6_traces.png")

plot_posterior(
    idata_m_5_6,
    hdi_prob=CI,
    var_names=["alpha", "beta_M", "sigma"],
    kind="hist",
    color="orangered",
    point_estimate="median",
)
savefig("m_5_6_posterior_hisograms")

plot_pair(idata_m_5_6, var_names=["alpha", "beta_M", "sigma"], kind="kde")
savefig("m_5_6_pairplot_alpha_beta_M_sigma.png")

# Linear model with milk energy content as a function of body and brain mass.
with Model() as m_5_7:
    """K_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_M * M_i + beta_N * N_i
    alpha ~ Normal(0, 0.2)
    beta_M ~ Normal(0, 0.5)
    beta_N ~ Normal(0, 0.5)
    sigma = Exponential(1)
    """
    # Priors.
    alpha = Normal("alpha", 0.0, 0.2)
    beta_M = Normal("beta_M", 0.0, 0.5)
    beta_N = Normal("beta_N", 0.0, 0.5)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu_i = Deterministic(
        "mu_i",
        alpha + beta_M * clean_milk_data["M"] + beta_N * clean_milk_data["N"],
    )
    K_i = Normal("K_i", mu_i, sigma, observed=clean_milk_data["K"])
    # Sampling and extracting inference data.
    prior_pc_m_5_7 = sample_prior_predictive()
    trace_m_5_7 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_5_7 = sample_posterior_predictive(trace_m_5_7)
    idata_m_5_7 = from_pymc3(
        trace_m_5_7,
        prior=prior_pc_m_5_7,
        posterior_predictive=post_pc_m_5_7,
        model=m_5_7,
    )

model_to_graphviz(m_5_7).render("m_5_7_dag", cleanup=True, format="png")

summary(idata_m_5_7, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_5_7_summary.csv"
)

plot_ppc(
    idata_m_5_7,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
)
savefig("m_5_7_posterior_pc.png")

plot_trace(
    idata_m_5_7, compact=True, var_names=["alpha", "beta_M", "beta_N", "sigma"]
)
savefig("m_5_7_traces.png")

plot_posterior(
    idata_m_5_7,
    hdi_prob=CI,
    var_names=["alpha", "beta_M", "beta_N", "sigma"],
    kind="hist",
    color="orangered",
    point_estimate="median",
)
savefig("m_5_7_posterior_hisograms")

plot_pair(
    idata_m_5_7, var_names=["alpha", "beta_M", "beta_N", "sigma"], kind="kde"
)
savefig("m_5_7_pairplot_alpha_beta_M_sigma.png")

# Compare the model parameters (c.f. DAG conditional indepencies).
plot_forest(
    [idata_m_5_5, idata_m_5_6, idata_m_5_7],
    model_names=["model 5.5", "model 5.6", "model 5.7"],
    var_names=["beta_M", "beta_N"],
    combined=True,
    hdi_prob=CI,
    colors=["black", "orangered", "cornflowerblue"],
)
savefig("forest_plot_beta_M_beta_N.png")

# Using DAGs to explain the forest plot.
# Directed acyclic graphs.
dag_0 = DAG([("M", "N"), ("M", "K"), ("N", "K")])
dag_0_plot = dag_0.to_daft(
    node_pos="circular", pgm_params={"observed_style": "inner"}
)
dag_0_plot.render()
dag_0_plot.savefig("milk_dag_0.png")

dag_1 = DAG([("N", "M"), ("M", "K"), ("N", "K")])
dag_1_plot = dag_1.to_daft(
    node_pos="circular", pgm_params={"observed_style": "inner"}
)
dag_1_plot.render()
dag_1_plot.savefig("milk_dag_1.png")

dag_2 = DAG([("M", "K"), ("N", "K")])
dag_2.add_node("U", latent=True)
dag_2.add_edge("U", "M")
dag_2.add_edge("U", "N")
dag_2_plot = dag_2.to_daft(
    node_pos="circular", pgm_params={"observed_style": "inner"}
)
dag_2_plot.render()
dag_2_plot.savefig("milk_dag_2.png")

# These are a Markov equivalent set (same conditional independencies in each.)
with open("milk_conditional_independencies.txt", "w") as output:
    output.write(
        f"DAG_0 conditional independencies:\n{dag_0.get_independencies()}\n"
    )
    output.write(
        f"DAG_1 conditional independencies:\n{dag_1.get_independencies()}\n"
    )
    output.write(
        f"DAG_2 conditional independencies:\n{dag_2.get_independencies()}"
    )
