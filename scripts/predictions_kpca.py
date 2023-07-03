"""
This script computes a clustering for the company description in the training data set and then accepts new descriptions
and assigns them to existing clusters. It uses KPCA for dimensionality reduction and K-Means clustering.
"""
import numpy as np
from code.description_data import train_descriptions, train_labels
from code.embeddings import get_description_embeddings
from code.clustering_kpca import fit_kernel_pca, cluster_data, predict_cluster, visualize_clustering

assert len(train_labels) == len(train_descriptions), print(len(train_labels), len(train_descriptions))

# Compute the clustering of the training data
train_embeddings = get_description_embeddings(train_descriptions)
n_clusters = int(np.floor(np.sqrt(len(train_descriptions)))) + 4
pca = fit_kernel_pca(data=train_embeddings, random_state=42)
clustering = cluster_data(data=train_embeddings, kpca=pca, n_clusters=n_clusters, random_state=42)
train_clusters, train_projected = predict_cluster(data=train_embeddings, kpca=pca, clustering=clustering)
visualize_clustering(train_clusters=train_clusters, train_labels=train_labels, train_projected=train_projected,
                     fontsize=8)

# Store clusters and projected embeddings in dictionaries for later use
clusters_dict = {}
for c_label in train_clusters:
    clusters_dict[c_label] = []
for e_label, c_label in zip(train_labels, train_clusters):
    clusters_dict[c_label].append(e_label)
train_projected_dict = {}
for train_label, projected in zip(train_labels, train_projected):
    train_projected_dict[train_label] = projected


print("Welcome to LM-based Saarland Economy Clustering!")
while True:
    print("Press 1 for a location recommendation and 0 for leaving the application.")
    command = int(input())
    if command == 0:
        break
    print("Please enter your company's name")
    test_label = input()
    print("Please enter your company's description")
    test_description = input()

    test_embeddings = get_description_embeddings([test_description])
    test_clusters, test_projected = predict_cluster(data=test_embeddings, kpca=pca, clustering=clustering)

    print("\n")
    print(f"Our analysis assigned {test_label} to the cluster {test_clusters[0]},"
          f" which contains the following companies (ordered according to similarity):")

    labels_and_distances = []
    for company in clusters_dict[test_clusters[0]]:
        company_projected = train_projected_dict[company]
        distance = np.linalg.norm(test_projected[0]-company_projected)
        labels_and_distances.append((company, distance))

    labels_and_distances.sort(key=lambda x: x[1])

    for i in range(len(labels_and_distances)):
        print(f"{i+1}: {labels_and_distances[i][0]} (similarity: {labels_and_distances[i][1]:.3f})")

    print("\n")
    print(f"Based on these results, we recommend locating {test_label} near {labels_and_distances[0][0]}.")
    print("If you would like to visualize our clustering results press 2 and if you would like to continue press 3.")
    command = int(input())
    if command == 2:
        visualize_clustering(train_clusters=train_clusters, train_labels=train_labels, train_projected=train_projected,
                             test_clusters=test_clusters, test_labels=[test_label], test_projected=test_projected,
                             fontsize=8)
    print("\n")
