from configparser import ConfigParser

from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork
from pgmpy.inference import CausalInference
from numpy import median
from numpy.random import binomial, normal
from pandas import DataFrame
from pymc3 import Model, Normal, Exponential, sample
from arviz import from_pymc3, summary

# Constants to be used later.
config = ConfigParser()
config.read("../config.ini")
SAMPLES = config.getint("parameters", "SAMPLES")
CHAINS = config.getint("parameters", "CHAINS")
PREDICTIVE_SAMPLES = config.getint("parameters", "PREDICTIVE_SAMPLES")
CI = config.getfloat("parameters", "CREDIBLE_INTERVAL")

# Create DAGs for model versions.
dag_gpc = DAG([("G", "P"), ("G", "C"), ("P", "C")])
dag_gpc_plot = dag_gpc.to_daft(
    node_pos="circular", pgm_params={"observed_style": "inner"}
)
dag_gpc_plot.render()
dag_gpc_plot.savefig("dag_gpc.png")
bn_gpc = BayesianNetwork(dag_gpc)

dag_gpcu = DAG([("G", "P"), ("G", "C"), ("P", "C")])
dag_gpcu.add_node("(u)", latent=True)
dag_gpcu.add_edge("(u)", "P")
dag_gpcu.add_edge("(u)", "C")
dag_gpcu_plot = dag_gpcu.to_daft(
    node_pos="circular", pgm_params={"observed_style": "inner"}
)
dag_gpcu_plot.render()
dag_gpcu_plot.savefig("dag_gpcu.png")
bn_gpcu = BayesianNetwork(dag_gpcu)

# Conditional independence.
with open("dag_gpc_conditional_independencies.txt", "w") as output:
    output.write(
        f"DAG_gpc conditional independencies:\n{dag_gpc.get_independencies()}\n"
    )

with open("dag_gpcu_conditional_independencies.txt", "w") as output:
    output.write(
        f"DAG_gpcu conditional independencies:\n{dag_gpcu.get_independencies()}\n"
    )

# Minimal adjustmnet set. No collider bias warning for gpcu...
inference_gpc = CausalInference(bn_gpc)
with open("bn_gpc_GC_adjustment.txt", "w") as output:
    output.write("GC adjustment set:\n")
    output.write(str(inference_gpc.get_minimal_adjustment_set("G", "C")))

inference_gpcu = CausalInference(bn_gpcu)
with open("bn_gpcu_PC_adjustment.txt", "w") as output:
    output.write("GC adjustment set:\n")
    output.write(str(inference_gpcu.get_minimal_adjustment_set("G", "C")))

# Simulate the GPCU DAG.
N = 200
b_GP = 1
b_GC = 0
b_PC = 1
b_U = 2
U = 2 * binomial(1, 0.5, N) - 1
G = normal(size=N)
P = normal(b_GP * G + b_U * U)
C = normal(b_PC * P + b_GC * G + b_U * U)
data = DataFrame.from_dict({"C": C, "P": P, "G": G, "U": U})

# Collider bias model with G and P.
with Model() as m_6_11:
    """C_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_PC * P_i + b_GC * G_i
    alpha ~ Normal(0, 1)
    beta_PC ~ Normal(0, 1)
    beta_GC ~ Normal(0, 1)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha = Normal("alpha", 0.0, 1.0)
    beta_PC = Normal("beta_PC", 0.0, 1.0)
    beta_GC = Normal("beta_GC", 0.0, 1.0)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu = alpha + beta_PC * data["P"] + beta_GC * data["G"]
    C = Normal("C", mu, sigma, observed=data["C"])
    # Sample.
    trace_m_6_11 = sample(SAMPLES, chains=CHAINS)
    idata_m_6_11 = from_pymc3(trace_m_6_11, model=m_6_11)

summary(idata_m_6_11, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_6_11_summary.csv"
)

# Adding in U, even though it couldn't be done normally.
with Model() as m_6_12:
    """C_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_PC * P_i + b_GC * G_i + b_U * U_i
    alpha ~ Normal(0, 1)
    beta_PC ~ Normal(0, 1)
    beta_GC ~ Normal(0, 1)
    beta_U ~ Normal(0, 1)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha = Normal("alpha", 0.0, 1.0)
    beta_PC = Normal("beta_PC", 0.0, 1.0)
    beta_GC = Normal("beta_GC", 0.0, 1.0)
    beta_U = Normal("beta_U", 0.0, 1.0)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu = alpha + beta_PC * data["P"] + beta_GC * data["G"] + beta_U * data["U"]
    C = Normal("C", mu, sigma, observed=data["C"])
    # Sample.
    trace_m_6_12 = sample(SAMPLES, chains=CHAINS)
    idata_m_6_12 = from_pymc3(trace_m_6_12, model=m_6_12)

summary(idata_m_6_12, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_6_12_summary.csv"
)
