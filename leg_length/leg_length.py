from numpy.random import normal, uniform

N = 100  # The number of individuals.
height = normal(10, 2, N)
leg_proportion = uniform(0.4, 0.5, N)
# Create leg lengths with Gaussian measurement error.
leg_length = leg_proportion * height
left_leg_length = leg_length + normal(0, 0.2, N)
right_leg_length = leg_length + normal(0, 0.2, N)
