import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regularized Linear Regression with Ridge and Lasso
alphas = np.logspace(-3, 2, 50)  # Range of alpha values
ridge_mse = []
lasso_mse = []

for alpha in alphas:
    # Ridge Regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    ridge_mse.append(mean_squared_error(y_test, y_pred_ridge))

    # Lasso Regression
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    lasso_mse.append(mean_squared_error(y_test, y_pred_lasso))


# Best Alpha for Ridge
best_alpha_ridge = alphas[np.argmin(ridge_mse)]
print(f"Best Alpha for Ridge: {best_alpha_ridge}")

# Best Alpha for Lasso
best_alpha_lasso = alphas[np.argmin(lasso_mse)]
print(f"Best Alpha for Lasso: {best_alpha_lasso}")

# Fit Ridge and Lasso with the best alpha values
ridge_best = Ridge(alpha=best_alpha_ridge)
ridge_best.fit(X_train, y_train)

lasso_best = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_best.fit(X_train, y_train)

print(f"Ridge Coefficients: {ridge_best.coef_}")
print(f"Lasso Coefficients: {lasso_best.coef_}")


# Plot MSE vs Alpha for Ridge and Lasso
plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_mse, label="Ridge", marker="o")
plt.plot(alphas, lasso_mse, label="Lasso", marker="s")
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge vs Lasso Regression')
plt.legend()
plt.grid(True)
plt.show()
