import numpy as np
from matrix_completion import svt_solve, pmf_solve, biased_mf_solve
import matplotlib.pyplot as plt


ratings_100k = np.load('rating-100k.npy')

plt.imshow(ratings_100k, cmap='jet', interpolation='nearest')
plt.show()

mask = (ratings_100k > 0).astype(np.int)

rHat_svt = svt_solve(ratings_100k, mask)
plt.imshow(rHat_svt, cmap='jet', interpolation='nearest')
plt.show()

rHat_pmf = pmf_solve(ratings_100k, mask, 10, 1e-2)
plt.imshow(rHat_pmf, cmap='jet', interpolation='nearest')
plt.show()

rHat_bias = biased_mf_solve(ratings_100k, mask, 10, 1e-2)
plt.imshow(rHat_bias, cmap='jet', interpolation='nearest')
plt.show()
