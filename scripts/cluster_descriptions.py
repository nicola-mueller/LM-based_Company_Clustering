"""
This script computes a clustering for given data and makes predictions for new data points.
"""
import numpy as np
from code.description_data import train_descriptions, train_labels, test_descriptions, test_labels
from code.embeddings import get_description_embeddings
from code.clustering import fit_pca, fit_kernel_pca, cluster_data, predict_cluster, visualize_clustering, \
    visualize_tsne, print_clusters

assert len(train_labels) == len(train_descriptions), print(len(train_labels), len(train_descriptions))

train_embeddings = get_description_embeddings(train_descriptions, embed_type="max")
test_embeddings = get_description_embeddings(test_descriptions, embed_type="max")

n_clusters = int(np.floor(np.sqrt(len(train_descriptions)))) + 4

pca = fit_kernel_pca(data=train_embeddings, dimensions=3)

clustering = cluster_data(data=train_embeddings, pca=pca, n_clusters=n_clusters, n_init=50,
                          init="random", random_state=69)

train_clusters, train_projected = predict_cluster(data=train_embeddings, pca=pca, clustering=clustering)
test_clusters, test_projected = predict_cluster(data=test_embeddings, pca=pca, clustering=clustering)

print_clusters(train_labels, train_clusters)

visualize_clustering(train_clusters=train_clusters, train_labels=train_labels, train_projected=train_projected,
                     test_clusters=test_clusters, test_labels=test_labels, test_projected=test_projected,
                     fontsize=8, cmap="rainbow", dimensions=3)

visualize_tsne(data=train_embeddings + test_embeddings, labels=train_labels + test_labels, dimensions=2,
               perplexity=np.sqrt(len(train_embeddings) + len(test_embeddings)), fontsize=5, random_state=69)
