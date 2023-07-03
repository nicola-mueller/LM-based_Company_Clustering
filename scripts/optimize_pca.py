"""
This script is used to find the optimal hyperparameters for PCA/KPCA via grid search.
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA, KernelPCA
from code.description_data import train_descriptions
from code.embeddings import get_description_embeddings

embeddings = get_description_embeddings(train_descriptions, embed_type="max")


def my_scorer(estimator, x):
    x_reduced = estimator.transform(x)
    x_preimage = estimator.inverse_transform(x_reduced)
    return -1 * mean_squared_error(x, x_preimage)


param_grid = [{
    "gamma": np.linspace(0.03, 0.05, 10),
    "kernel": ["rbf", "sigmoid", "linear", "poly"],
    "degree": [2, 3, 4, 5]
}]
kpca = KernelPCA(fit_inverse_transform=True, n_components=3)
grid_search = GridSearchCV(kpca, param_grid, cv=3, scoring=my_scorer, n_jobs=-1)
grid_search.fit(embeddings)
print(grid_search.cv_results_)
print(grid_search.best_params_)
