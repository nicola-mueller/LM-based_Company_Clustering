"""
This script contains all the functions necessary for clustering data using t-SNE and Agglomerative clustering.
"""
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull
from scipy import interpolate
import matplotlib

colors = matplotlib.colors.ListedColormap(
    [
        "rosybrown",
        "darkred",
        "tan",
        "darkkhaki",
        "forestgreen",
        "teal",
        "steelblue",
        "rebeccapurple",
        "magenta",
        "crimson",
        "silver",
        "brown",
        "orange",
        "darkgoldenrod",
        "olive",
        "lime",
        "aqua",
        "royalblue",
        "darkorchid",
        "hotpink"
    ]
).colors

colors = matplotlib.colors.ListedColormap(plt.cm.tab20.colors).colors


# Prints the results of clustering
def print_clusters(data_labels, clustering_labels):
    clusters = {}
    for c_label in clustering_labels:
        clusters[c_label] = []

    for e_label, c_label in zip(data_labels, clustering_labels):
        clusters[c_label].append(e_label)

    for c_label in clusters.keys():
        print("Cluster " + str(c_label) + ":")
        print(clusters[c_label])
        print("\n")


# Fits an instance of t-SNE using given data
def project_tsne(data, dimensions, perplexity, random_state=69):
    projected_data = TSNE(n_components=dimensions, perplexity=perplexity, random_state=random_state).fit_transform(
        np.array(data))

    return projected_data


# Creates a dendrogram to decide the number of clusters
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)


# Computes clusters
def predict_clusters(projected_data, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(projected_data)
    clustering_labels = clustering.labels_

    return clustering_labels, projected_data


# Visualizes the results of clustering as 2D scatter plots
def visualize_clustering(clusters, labels, projected, fontsize=5):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot()
    ax.set_title("Agglomerative Clustering of Description Embeddings Projected into 2D Space Using t-SNE",
                 fontsize=fontsize + 10)

    # plot data points and names
    for i, txt in enumerate(labels):
        ax.scatter(projected[i, 0], projected[i, 1], color=colors[clusters[i]])
        ax.text(projected[i, 0], projected[i, 1], txt, fontsize=fontsize)

    # draw shapes around clusters
    for cluster in np.unique(clusters):
        points = projected[clusters == cluster]

        if len(points) == 2:
            # create fake points to compute convex hull
            if abs(points[0, 0] - points[1, 0]) < 1:
                # same x
                fake_point_1 = np.array(
                    [np.mean(points[:, 0]), points[0, 1] + (np.max(points[:, 0] - np.mean(points[:, 0])))])
                fake_point_2 = np.array(
                    [np.mean(points[:, 0]), points[0, 1] - (np.max(points[:, 0] - np.mean(points[:, 0])))])
            elif abs(points[0, 1] - points[1, 1]) < 1:
                # same y
                fake_point_1 = np.array(
                    [points[0, 0] + (np.max(points[:, 1] - np.mean(points[:, 1]))), np.mean(points[:, 1])])
                fake_point_2 = np.array(
                    [points[0, 0] - (np.max(points[:, 1] - np.mean(points[:, 1]))), np.mean(points[:, 1])])
            else:
                fake_point_1 = np.array([np.min(points[:, 0]), np.min(points[:, 1])])
                fake_point_2 = np.array([np.max(points[:, 0]), np.max(points[:, 1])])
            points = np.append(points, fake_point_1).reshape(-1, 2)
            points = np.append(points, fake_point_2).reshape(-1, 2)
        elif len(points) == 1:
            # just write cluster name since we cant draw a shape
            offset = (np.max(projected[:, 1]) - np.min(projected[:, 1])) * 0.025
            ax.text(np.min(points[:, 0]), np.max(points[:, 1]) + offset, "Cluster " + str(cluster),
                    fontsize=fontsize + 5, alpha=0.5, color=colors[cluster])
            continue

        # compute convex hull
        try:
            hull = ConvexHull(
                points)  # computation might not work if data points in a cluster have a very weird position
        except Exception as e:
            print(e)
            # plot the label of the cluster
            offset = (np.max(projected[:, 1]) - np.min(projected[:, 1])) * 0.025
            ax.text(np.min(points[:, 0]), np.max(points[:, 1]) + offset, "Cluster " + str(cluster),
                    fontsize=fontsize + 5, alpha=0.5, color=colors[cluster])
            continue
        x_hull = np.append(points[hull.vertices, 0],
                           points[hull.vertices, 0][0])
        y_hull = np.append(points[hull.vertices, 1],
                           points[hull.vertices, 1][0])

        # interpolate to get a smooth shape
        dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
        plt.fill(interp_x, interp_y, '--', c=colors[cluster], alpha=0.2)

        # plot the label of the cluster
        offset = (np.max(projected[:, 1]) - np.min(projected[:, 1])) * 0.025
        ax.text(np.min(points[:, 0]), np.max(points[:, 1]) + offset, "Cluster " + str(cluster), fontsize=fontsize + 5,
                alpha=0.5, color=colors[cluster])

    # plt.savefig("clustering_tsne.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show(block=True)
