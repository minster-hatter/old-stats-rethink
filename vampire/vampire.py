# Based on code 3.1 in Statistical Rethinking - P(vampire|+).
P_POSITIVE_VAMPIRE = 0.95
P_POSITIVE_MORTAL = 0.01
P_VAMPIRE = 0.001
P_POSITIVE = P_POSITIVE_VAMPIRE * P_VAMPIRE + P_POSITIVE_MORTAL * (
    1 - P_VAMPIRE
)
P_VAMPIRE_POSITIVE = P_VAMPIRE * P_POSITIVE_VAMPIRE / (P_POSITIVE)
print(
    "The probability of being a vampire, given a positive test result, is "
    f"{round(P_VAMPIRE_POSITIVE, 3)}."
)
