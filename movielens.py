import numpy as np
from matrix_completion import svt_solve, calc_unobserved_rmse

U = np.random.randn(20, 5) # user latent factors
V = np.random.randn(15, 5) # item latent factors
R = np.random.randn(20, 15) + np.dot(U, V.T) # R = UV^T + noise

mask = np.round(np.random.rand(20, 15)) # 1 = seen, 0 = hidden
R_hat = svt_solve(R, mask)

print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
