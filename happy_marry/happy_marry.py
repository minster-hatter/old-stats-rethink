from configparser import ConfigParser

from numpy import exp, zeros, repeat, arange, linspace, array, median
from numpy.random import seed, binomial
from pandas import DataFrame, Categorical
from arviz import summary, from_pymc3
from pymc3 import Model, Normal, Exponential, sample, model_to_graphviz
from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork
from pgmpy.inference import CausalInference

# Constants to be used later.
config = ConfigParser()
config.read("../config.ini")
SAMPLES = config.getint("parameters", "SAMPLES")
CHAINS = config.getint("parameters", "CHAINS")
PREDICTIVE_SAMPLES = config.getint("parameters", "PREDICTIVE_SAMPLES")
CI = config.getfloat("parameters", "CREDIBLE_INTERVAL")

COLUMN_NAMES = ["age", "happiness", "married"]


def inv_logit(x):
    return exp(x) / (1 + exp(x))


def sim_happiness(n_years=100, seed_number=1234):
    seed(seed_number)
    popn = DataFrame(zeros((20 * 65, 3)), columns=COLUMN_NAMES)
    popn.loc[:, "age"] = repeat(arange(65), 20)
    popn.loc[:, "happiness"] = repeat(linspace(-2, 2, 20), 65)
    popn.loc[:, "married"] = array(popn.loc[:, "married"].values, dtype="bool")
    for i in range(n_years):
        # Make the population age by a year.
        popn.loc[:, "age"] += 1
        # Replace over 65s with new people.
        ind = popn["age"] == 65
        popn.loc[ind, "age"] = 0
        popn.loc[ind, "married"] = False
        popn.loc[ind, "happiness"] = linspace(-2, 2, 20)
        # Make some people marry.
        elligible = (popn.married == 0) & (popn.age >= 18)
        marry = (
            binomial(1, inv_logit(popn.loc[elligible, "happiness"] - 4)) == 1
        )
        popn.loc[elligible, "married"] = marry
    popn.sort_values("age", inplace=True, ignore_index=True)
    return popn


population = sim_happiness()
population_summary = population.copy()
# Make booleans into integers for arviz.
population_summary["married"] = population_summary["married"].astype(int)
summary(
    population_summary.to_dict(orient="list"), kind="stats", hdi_prob=CI
).to_csv("population_summary.csv")

# Focus only on adults and rescale age 18 to 65 as 0 to 1.
adults = population.loc[population.age >= 18]
adults.loc[:, "A"] = (adults["age"].copy() - 18) / (65 - 18)

married_id = Categorical(adults.loc[:, "married"].astype(int))

# Model happiness as a function age and married.
with Model() as m_6_9:
    """H_i ~ Normal(mu_i, sigma)
    mu_i = alpha_m[i] + beta_A * A_i
    alpha_m[j] ~ Normal(0, 1)
    beta_A ~ Normal(0, 2)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha_m = Normal("alpha_m", 0.0, 1.0, shape=2)
    beta_A = Normal("beta_A", 0.0, 2.0)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu = alpha_m[married_id] + beta_A * adults["A"]
    happiness = Normal("H", mu, sigma, observed=adults["happiness"])
    # Sample
    m_6_9_trace = sample(SAMPLES, chains=CHAINS)
    idata_m_6_9 = from_pymc3(m_6_9_trace, model=m_6_9)

model_to_graphviz(m_6_9).render("m_6_9_dag", cleanup=True, format="png")

summary(idata_m_6_9, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_6_9_summary.csv"
)

# A model without marriage status shows (correctly) no relationship.
with Model() as m_6_10:
    """H_i ~ Normal(mu_i, sigma)
    mu_i = alpha + beta_A * A_i
    alpha ~ Normal(0, 1)
    beta_A ~ Normal(0, 2)
    sigma ~ Exponential(1)
    """
    # Priors.
    alpha = Normal("alpha", 0.0, 1.0)
    beta_A = Normal("beta_A", 0.0, 2.0)
    sigma = Exponential("sigma", 1.0)
    # Likelihood.
    mu = alpha + beta_A * adults["A"]
    happiness = Normal("H", mu, sigma, observed=adults["happiness"])
    # Sample
    m_6_10_trace = sample(SAMPLES, chains=CHAINS)
    idata_m_6_10 = from_pymc3(m_6_10_trace, model=m_6_10)

model_to_graphviz(m_6_10).render("m_6_10_dag", cleanup=True, format="png")

summary(idata_m_6_10, hdi_prob=CI, stat_funcs=[median]).to_csv(
    "m_6_10_summary.csv"
)

# DAG and implications.
dag = DAG([("H", "M"), ("A", "M")])
dag_plot = dag.to_daft(
    node_pos="circular", pgm_params={"observed_style": "inner"}
)
dag_plot.render()
dag_plot.savefig("marry_dag.png")

with open("marriage_conditional_independencies.txt", "w") as output:
    output.write(
        f"marriage DAG conditional independencies:\n{dag.get_independencies()}\n"
    )

bn = BayesianNetwork(dag)
inference = CausalInference(bn)
with open("marriage_AH_adjustment.txt", "w") as output:
    output.write("AH adjustment set:\n")
    output.write(str(inference.get_minimal_adjustment_set("A", "H")))
