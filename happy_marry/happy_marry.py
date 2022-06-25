from configparser import ConfigParser

from numpy import exp, zeros, repeat, arange, linspace, array
from numpy.random import seed, binomial
from pandas import DataFrame
from arviz import summary

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
print(summary(population_summary.to_dict(orient="list"), kind="stats", hdi_prob=CI))
