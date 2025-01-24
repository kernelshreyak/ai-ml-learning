import numpy as np
from scipy.stats import chi2_contingency

# Observed frequency table
observed = np.array([[30, 20],
                     [20, 30]])

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(observed)

# Results
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis: Variables are dependent.")
else:
    print("Fail to reject the null hypothesis: Variables are independent.")
