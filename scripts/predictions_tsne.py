"""
This script computes a clustering for the company description in the training data set and then accepts new descriptions
and assigns them to existing clusters. It uses t-SNE for dimensionality reduction and Agglomerative clustering.
"""
import numpy as np
from code.description_data import train_descriptions, train_labels
from code.embeddings import get_description_embeddings
from code.clustering_tsne import project_tsne, predict_clusters, visualize_clustering

assert len(train_labels) == len(train_descriptions), print(len(train_labels), len(train_descriptions))

# Compute the clustering of the training data
train_embeddings = get_description_embeddings(train_descriptions)
n_clusters = int(np.floor(np.sqrt(len(train_descriptions)))) + 5
projected_train_embeddings = project_tsne(data=train_embeddings, dimensions=2,
                                          perplexity=np.sqrt(len(train_embeddings)),
                                          random_state=42)
clusters, projected = predict_clusters(projected_data=projected_train_embeddings, n_clusters=n_clusters)

visualize_clustering(clusters=clusters, labels=train_labels, projected=projected,
                     fontsize=8)

print("Welcome to LM-based Saarland Economy Clustering!")
while True:
    print("Press 1 for a location recommendation and 0 for leaving the application.")
    command = int(input())
    if command == 0:
        break
    print("Please enter your company's name")
    new_label = input()
    print("Please enter your company's description")
    new_description = input()

    new_embedding = get_description_embeddings([new_description])
    all_embeddings = train_embeddings + new_embedding
    all_labels = train_labels + [new_label]
    projected_embeddings = project_tsne(data=all_embeddings, dimensions=2, perplexity=np.sqrt(len(all_embeddings)),
                                        random_state=69)
    clusters, projected = predict_clusters(projected_data=projected_embeddings, n_clusters=n_clusters)

    print("\n")
    print(
        f"Our analysis assigned {new_label} to the cluster {clusters[-1]}, which contains"
        f" the following companies (ordered according to similarity):")

    clusters_dict = {}
    for c_label in clusters:
        clusters_dict[c_label] = []
    for e_label, c_label in zip(all_labels, clusters):
        clusters_dict[c_label].append(e_label)
    projected_dict = {}
    for train_label, project in zip(all_labels, projected):
        projected_dict[train_label] = project

    labels_and_distances = []
    new_projected = projected_dict[all_labels[-1]]
    for company in clusters_dict[clusters[-1]]:
        if company != new_label:
            company_projected = projected_dict[company]
            distance = np.linalg.norm(new_projected - company_projected)
            labels_and_distances.append((company, distance))

    labels_and_distances.sort(key=lambda x: x[1])

    for i in range(len(labels_and_distances)):
        print(f"{i + 1}: {labels_and_distances[i][0]} (similarity: {labels_and_distances[i][1]:.3f})")

    print("\n")
    print(f"Based on these results, we recommend locating {new_label} near {labels_and_distances[0][0]}.")
    print("If you would like to visualize our clustering results press 2 and if you would like to continue press 3.")
    command = int(input())
    if command == 2:
        visualize_clustering(clusters=clusters, labels=all_labels, projected=projected,
                             fontsize=8)
    print("\n")
