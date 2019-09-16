import numpy as np
from matrix_completion import svt_solve, pmf_solve, biased_mf_solve
from custom_rmse import calc_unobserved_rmse, calc_observed_rmse
import matplotlib.pyplot as plt

users = 200
items = 150
ranks = np.arange(5, 41, 5)
percent_shown = 0.3
res_observed = np.zeros((np.size(ranks), 3))
res_unobserved = np.zeros((np.size(ranks), 3))

for i in range(len(ranks)):
    U = np.random.randn(users, ranks[i])
    V = np.random.randn(items, ranks[i])
    # R = np.random.randn(users, items) + np.dot(U, V.T) # with noise
    R = np.dot(U, V.T) # without noise

    mask = np.random.binomial(1, percent_shown, size=users*items).reshape((users, items))

    rHat_svt = svt_solve(R, mask)
    res_observed[i, 0] = calc_observed_rmse(R, rHat_svt, mask)
    res_unobserved[i, 0] = calc_unobserved_rmse(R, rHat_svt, mask)

    rHat_pmf = pmf_solve(R, mask, 10, 1e-2)
    res_observed[i, 1] = calc_observed_rmse(R, rHat_pmf, mask)
    res_unobserved[i, 1] = calc_unobserved_rmse(R, rHat_pmf, mask)

    rHat_bias = biased_mf_solve(R, mask, 10, 1e-2)
    res_observed[i, 2] = calc_observed_rmse(R, rHat_bias, mask)
    res_unobserved[i, 2] = calc_unobserved_rmse(R, rHat_bias, mask)

plt.plot(ranks, res_observed[:, 0], 'r--', label='SVT')
plt.plot(ranks, res_observed[:, 1], 'g--', label='PMF')
plt.plot(ranks, res_observed[:, 2], 'b--', label='BIAS')
plt.title('Observed RMSE')
plt.legend(loc='upper left')
plt.show()

plt.plot(ranks, res_unobserved[:, 0], 'r--', label='SVT')
plt.plot(ranks, res_unobserved[:, 1], 'g--', label='PMF')
plt.plot(ranks, res_unobserved[:, 2], 'b--', label='BIAS')
plt.title('UnObserved RMSE')
plt.legend(loc='upper left')
plt.show()
