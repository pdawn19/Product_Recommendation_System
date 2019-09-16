import numpy as np
from matrix_completion import svt_solve, pmf_solve, biased_mf_solve
from custom_rmse import calc_unobserved_rmse, calc_observed_rmse
import matplotlib.pyplot as plt

users = 200
items = 150
rank = 10
percent_shown = 0.5

U = np.random.randn(users, rank)
V = np.random.randn(items, rank)
# R = np.random.randn(users, items) + np.dot(U, V.T) # with noise
R = np.dot(U, V.T) # without noise
plt.imshow(R, cmap='viridis', interpolation='nearest')
plt.title('Full Data')
plt.colorbar()
plt.show()

mask = np.random.binomial(1, percent_shown, size=users*items).reshape((users, items))
plt.imshow(mask, cmap='viridis', interpolation='nearest')
plt.title('Mask')
plt.colorbar()
plt.show()

rHat_svt = svt_solve(R, mask)
print('SVT Solve')
print('Observed RMSE:', calc_observed_rmse(R, rHat_svt, mask))
print('UnObserved RMSE:', calc_unobserved_rmse(R, rHat_svt, mask))
plt.imshow(rHat_svt, cmap='viridis', interpolation='nearest')
plt.title('SVT')
plt.colorbar()
plt.show()

rHat_pmf = pmf_solve(R, mask, 10, 1e-2)
print('PMF Solve')
print('Observed RMSE:', calc_observed_rmse(R, rHat_pmf, mask))
print('UnObserved RMSE:', calc_unobserved_rmse(R, rHat_pmf, mask))
plt.imshow(rHat_pmf, cmap='viridis', interpolation='nearest')
plt.title('PMF')
plt.colorbar()
plt.show()

rHat_bias = biased_mf_solve(R, mask, 10, 1e-2)
print('Bias Solve')
print('Observed RMSE:', calc_observed_rmse(R, rHat_bias, mask))
print('UnObserved RMSE:', calc_unobserved_rmse(R, rHat_bias, mask))
plt.imshow(rHat_bias, cmap='viridis', interpolation='nearest')
plt.title('Bias')
plt.colorbar()
plt.show()
