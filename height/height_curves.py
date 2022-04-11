# Polynomials linking height and weight that are later compared.
from pathlib import Path
from configparser import ConfigParser
from sqlite3 import connect

from pandas import read_sql
from pymc3 import Model, Normal, Lognormal, Uniform, sample, model_to_graphviz
from arviz import from_pymc3, summary
from numpy import median, vstack, linspace
from numpy.random import randint
from matplotlib.pyplot import subplots, xlabel, ylabel, savefig

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
howell_data = read_sql("SELECT * FROM howell1", cnxn)

# Standardized predictors aids interpretability of squared (etc.) values.
howell_data["s_weight"] = (
    howell_data["weight"] - howell_data["weight"].mean()
) / howell_data["weight"].std()
howell_data["s_weight_squared"] = howell_data["s_weight"] ** 2
howell_data["s_weight_cubed"] = howell_data["s_weight"] ** 3


# Linear model.
with Model() as m_4_4:
    """h_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta * x_i
    alpha ~ Normal(178, 20)
    beta ~ Normal(0, 1)
    sigma ~ Uniform(0, 50)
    """
    # Priors.
    alpha = Normal("alpha", 178.0, 20.0)
    beta = Normal("beta", 0.0, 1.0)
    sigma = Uniform("sigma", 0.0, 50.0)
    # Likelihood.
    mu_i = alpha + beta * howell_data["s_weight"]
    h_i = Normal("h_i", mu_i, sigma, observed=howell_data["height"])
    # Sample and extract.
    trace_m_4_4 = sample(SAMPLES, chains=CHAINS)
    idata_m_4_4 = from_pymc3(trace_m_4_4, model=m_4_4)

model_to_graphviz(m_4_4).render("m_4_4_dag", cleanup=True, format="png")

summary(idata_m_4_4, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_4_4_summary.csv"
)

# Quadratic model.
with Model() as m_4_5:
    """h_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_1 * x_i + beta_2 * (x_i)^2
    alpha ~ Normal(178, 20)
    beta_1 ~ Log-Normal(0, 1)
    beta_2 ~ Normal(0, 1)
    sigma ~ Uniform(0, 50)
    """
    # Priors.
    alpha = Normal("alpha", 178.0, 20.0)
    beta_1 = Lognormal("beta_1", 0.0, 1.0)
    beta_2 = Normal("beta_2", 0.0, 1.0)
    sigma = Uniform("sigma", 0.0, 50.0)
    # Likelihood.
    mu_i = (
        alpha
        + beta_1 * howell_data["s_weight"]
        + beta_2 * howell_data["s_weight_squared"]
    )
    h_i = Normal("h_i", mu_i, sigma, observed=howell_data["height"])
    # Sample and extract.
    trace_m_4_5 = sample(SAMPLES, chains=CHAINS)
    idata_m_4_5 = from_pymc3(trace_m_4_5, model=m_4_5)

model_to_graphviz(m_4_5).render("m_4_5_dag", cleanup=True, format="png")

summary(idata_m_4_5, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_4_5_summary.csv"
)

# Cubic model.

# Make a matrix of weight predictor values for coding convenience.
with Model() as m_4_6:
    """h_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_1 * x_i + beta_2 * (x_i)^2 + beta_3 * (x_i)^3
    alpha ~ Normal(178, 20)
    beta_1 ~ Log-Normal(0, 1)
    beta_2 ~ Normal(0, 1)
    beta_3 ~ Normal(0, 1)
    sigma ~ Uniform(0, 50)
    """
    # Priors.
    alpha = Normal("alpha", 178.0, 20.0)
    beta_1 = Lognormal("beta_1", 0.0, 1.0)
    beta_2 = Normal("beta_2", 0.0, 1.0)
    beta_3 = Normal("beta_3", 0.0, 1.0)
    sigma = Uniform("sigma", 0.0, 50.0)
    # Likelihood.
    mu_i = (
        alpha
        + beta_1 * howell_data["s_weight"]
        + beta_2 * howell_data["s_weight_squared"]
        + beta_3 * howell_data["s_weight_cubed"]
    )
    h_i = Normal("h_i", mu_i, sigma, observed=howell_data["height"])
    # Sample and extract.
    trace_m_4_6 = sample(SAMPLES, chains=CHAINS)
    idata_m_4_6 = from_pymc3(trace_m_4_6, model=m_4_6)

model_to_graphviz(m_4_6).render("m_4_6_dag", cleanup=True, format="png")

summary(idata_m_4_6, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_4_6_summary.csv"
)

# Plot the models against the data.
random_indices = randint(len(trace_m_4_4), size=25)
XS = linspace(-3, 3, len(howell_data))
fig, ax = subplots(3, 1, sharex=True)
ax[0].scatter(
    howell_data["s_weight"], howell_data["height"], marker="x", color="black"
)
for index in random_indices:
    ax[0].plot(
        XS,
        trace_m_4_4["alpha"][index] + trace_m_4_4["beta"][index] * XS,
        "orangered",
        alpha=0.1,
    )
ax[0].set_ylabel("Height (cm)")
ax[1].scatter(
    howell_data["s_weight"], howell_data["height"], marker="x", color="black"
)
ax[1].set_ylabel("Height (cm)")
for index in random_indices:
    ax[1].plot(
        XS,
        trace_m_4_5["alpha"][index]
        + trace_m_4_5["beta_1"][index] * XS
        + trace_m_4_5["beta_2"][index] * XS ** 2,
        "orangered",
        alpha=0.1,
    )
ax[2].scatter(
    howell_data["s_weight"], howell_data["height"], marker="x", color="black"
)
for index in random_indices:
    ax[2].plot(
        XS,
        trace_m_4_6["alpha"][index]
        + trace_m_4_6["beta_1"][index] * XS
        + trace_m_4_6["beta_2"][index] * XS ** 2
        + trace_m_4_6["beta_3"][index] * XS ** 3,
        "orangered",
        alpha=0.1,
    )
ax[2].set_ylabel("Height (cm)")
xlabel("Weight (standardized)")
savefig("models_4_4_to_6_models.png")
