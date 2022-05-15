from numpy import quantile, corrcoef
from numpy.random import normal

N = 200  # The number of grant proposals.
P = 0.1  # The proportion of proposals that get accepted.
news = normal(size=N)
trust = normal(size=N)
# Simulate the selection process by finding the top 10% of combined scores.
score = news + trust
accpeted_score = quantile(score, 1 - P)
selected = s >= accepted_score
# Find that the correlation coefficient of truly uncorrelated values now ...
correlation_coef = corrcoef(news[selected], trust[selected])
print(f"The apparent correlation is {correlation_coef}.")
