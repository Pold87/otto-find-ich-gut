from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
X = pd.read_csv('data/train80.csv').iloc[:, 1:-1]
pca = PCA(n_components= 'mle')
pca.fit_transform(X)

print(pca.get_covariance())