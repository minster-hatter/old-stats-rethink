from pathlib import Path
from configparser import ConfigParser
from sqlite3 import connect

from pandas import read_sql
from sklearn.preprocessing import scale
from pymc3 import (
    Model,
    Normal,
    Exponential,
    sample_prior_predictive,
    model_to_graphviz,
    sample,
    sample_posterior_predictive,
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
from numpy import median
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
waffles_data = read_sql("SELECT * FROM waffles", cnxn)
print(waffles_data.describe())

# Standardize the divorce and median age at marriage fields.
waffles_data["D"] = scale(waffles_data["Divorce"])
waffles_data["A"] = scale(waffles_data["MedianAgeMarriage"])
waffles_data["M"] = scale(waffles_data["Marriage"])

# Model A as a linear predictor of D.
with Model() as m_5_1:
    """D_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_A * A_i
    alpha ~ Normal(0, 0.2)
    beta_A ~ Normal(0, 0.5)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha = Normal("alpha", 0.0, 0.2)
    beta_A = Normal("beta_A", 0.0, 0.5)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu_i = alpha + beta_A * waffles_data["A"]
    D_i = Normal("D_i", mu_i, sigma, observed=waffles_data["D"])
    # Prior check.
    prior_pc_m_5_1 = sample_prior_predictive()
    idata_m_5_1 = from_pymc3(prior=prior_pc_m_5_1, model=m_5_1)

model_to_graphviz(m_5_1).render("m_5_1_dag", cleanup=True, format="png")

plot_ppc(
    idata_m_5_1,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    group="prior",
    kind="cumulative",
)
savefig("m_5_1_prior_pc.png")

with m_5_1:
    trace_m_5_1 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_5_1 = sample_posterior_predictive(trace_m_5_1)
    idata_m_5_1 = from_pymc3(
        trace_m_5_1,
        prior=prior_pc_m_5_1,
        posterior_predictive=post_pc_m_5_1,
        model=m_5_1,
    )

summary(idata_m_5_1, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_5_1_summary.csv"
)

plot_ppc(
    idata_m_5_1,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
)
savefig("m_5_1_posterior_pc.png")

plot_trace(idata_m_5_1, compact=True, var_names=["alpha", "beta_A", "sigma"])
savefig("m_5_1_traces.png")

plot_posterior(
    idata_m_5_1,
    hdi_prob=CI,
    var_names=["alpha", "beta_A", "sigma"],
    kind="hist",
    color="orangered",
    point_estimate="median",
)
savefig("m_5_1_posterior_hisograms")

plot_pair(idata_m_5_1, var_names=["alpha", "beta_A"], kind="kde")
savefig("m_5_1_pairplot_alpha_beta_A.png")

# Model M as a linear predictor of D.
with Model() as m_5_2:
    """D_i ~ Normal(mu_i, sigma)
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
    mu_i = alpha + beta_M * waffles_data["M"]
    D_i = Normal("D_i", mu_i, sigma, observed=waffles_data["D"])
    # Sampling.
    prior_pc_m_5_2 = sample_prior_predictive()
    trace_m_5_2 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_5_2 = sample_posterior_predictive(trace_m_5_2)
    idata_m_5_2 = from_pymc3(
        trace_m_5_2,
        prior=prior_pc_m_5_2,
        posterior_predictive=post_pc_m_5_2,
        model=m_5_2,
    )
model_to_graphviz(m_5_2).render("m_5_2_dag", cleanup=True, format="png")

plot_ppc(
    idata_m_5_2,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    group="prior",
    kind="cumulative",
)
savefig("m_5_2_prior_pc.png")

summary(idata_m_5_2, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_5_2_summary.csv"
)

plot_ppc(
    idata_m_5_2,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
)
savefig("m_5_2_posterior_pc.png")

plot_trace(idata_m_5_2, compact=True, var_names=["alpha", "beta_M", "sigma"])
savefig("m_5_2_traces.png")

plot_posterior(
    idata_m_5_2,
    hdi_prob=CI,
    var_names=["alpha", "beta_M", "sigma"],
    kind="hist",
    color="orangered",
    point_estimate="median",
)
savefig("m_5_2_posterior_hisograms")

plot_pair(idata_m_5_2, var_names=["alpha", "beta_M"], kind="kde")
savefig("m_5_2_pairplot_alpha_beta_M.png")

# Directed acyclic graphs.
dag_0 = DAG([("A", "M"), ("A", "D"), ("M", "D")])
dag_0_plot = dag_0.to_daft(
    node_pos="circular", pgm_params={"observed_style": "inner"}
)
dag_0_plot.render()
dag_0_plot.savefig("waffles_dag_0.png")


dag_1 = DAG([("A", "M"), ("A", "D")])
dag_1_plot = dag_1.to_daft(
    node_pos="circular", pgm_params={"observed_style": "inner"}
)
dag_1_plot.render()
dag_1_plot.savefig("waffles_dag_1.png")

# Identift conditional independencies in DAGs.
with open("conditional_independencies.txt", "w") as output:
    output.write(
        f"DAG_0 conditional independencies:\n{dag_0.get_independencies()}\n"
    )
    output.write(
        f"DAG_1 conditional independencies:\n{dag_1.get_independencies()}"
    )

# Multiple regression model to test DAG implications.
with Model() as m_5_3:
    """D_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_M * M_i + beta_A * A_i
    alpha ~ Normal(0, 0.2)
    beta_M ~ Normal(0, 0,5)
    beta_A ~ Normal(0, 0,5)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha = Normal("alpha", 0.0, 0.2)
    beta_M = Normal("beta_M", 0.0, 0.5)
    beta_A = Normal("beta_A", 0.0, 0.5)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu_i = alpha + beta_M * waffles_data["M"] + beta_A * waffles_data["A"]
    D_i = Normal("D_i", mu_i, sigma, observed=waffles_data["D"])
    # Sampling.
    prior_pc_m_5_3 = sample_prior_predictive()
    trace_m_5_3 = sample(SAMPLES, chains=CHAINS)
    post_pc_m_5_3 = sample_posterior_predictive(trace_m_5_3)
    idata_m_5_3 = from_pymc3(
        trace_m_5_3,
        prior=prior_pc_m_5_3,
        posterior_predictive=post_pc_m_5_3,
        model=m_5_3,
    )
model_to_graphviz(m_5_3).render("m_5_3_dag", cleanup=True, format="png")

plot_ppc(
    idata_m_5_3,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    group="prior",
    kind="cumulative",
)
savefig("m_5_3_prior_pc.png")

summary(idata_m_5_3, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_5_3_summary.csv"
)

plot_ppc(
    idata_m_5_3,
    num_pp_samples=PREDICTIVE_SAMPLES,
    mean=False,
    kind="cumulative",
)
savefig("m_5_3_posterior_pc.png")

plot_trace(
    idata_m_5_3, compact=True, var_names=["alpha", "beta_M", "beta_A", "sigma"]
)
savefig("m_5_3_traces.png")

plot_posterior(
    idata_m_5_3,
    hdi_prob=CI,
    var_names=["alpha", "beta_M", "beta_A", "sigma"],
    kind="hist",
    color="orangered",
    point_estimate="median",
)
savefig("m_5_3_posterior_hisograms")

plot_pair(idata_m_5_3, var_names=["alpha", "beta_M", "beta_A"], kind="kde")
savefig("m_5_3_pairplot_alpha_beta_M_beta_A.png")

# Compare the model parameters (c.f. DAG conditional indepencies).
plot_forest(
    [idata_m_5_1, idata_m_5_2, idata_m_5_3],
    model_names=["model 5.1", "model 5.2", "model 5.3"],
    var_names=["beta_A", "beta_M"],
    combined=True,
    hdi_prob=CI,
    colors=["black", "orangered", "cornflowerblue"],
)
savefig("forest_plot_beta_A_beta_M.png")
