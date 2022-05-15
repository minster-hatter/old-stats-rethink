from numpy import quantile, corrcoef
from numpy.random import seed, normal

# Correlation arising from selection (see start of Chapter 6).
seed(5)
N = 200  # The number of grant proposals.
P = 0.1  # The proportion of proposals that get accepted.
news = normal(size=N)
trust = normal(size=N)
# Simulate the selection process by finding the top 10% of combined scores.
score = news + trust
accepted_score = quantile(score, 1 - P)
selected = score >= accepted_score
# Find that the correlation coefficient of truly uncorrelated values now ...
correlation_coef = corrcoef(news[selected], trust[selected])
rounded_correlation_coef = round(correlation_coef[0][1], 2)
print(f"The apparent correlation is {rounded_correlation_coef}.")
