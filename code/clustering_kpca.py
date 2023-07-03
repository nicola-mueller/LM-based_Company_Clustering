"""
This script contains all the functions necessary for clustering data using KPCA and K-Means.
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
import numpy as np
from scipy.spatial import ConvexHull
from scipy import interpolate
import matplotlib

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


# Fits an instance of Kernel Principal Components Analysis using given data
def fit_kernel_pca(data, kernel='poly', gamma=0.05, random_state=69, n_components=2):
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, random_state=random_state)
    kpca.fit(data)

    return kpca


# Computes the clustering: First the data's dimensionality is reduced by transforming it according to the given PCA/KPCA
# instance and then K-Means is applied to cluster the proejcted data
def cluster_data(data, kpca, n_clusters, random_state=69):
    projected_data = kpca.transform(data)
    clustering = KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state).fit(projected_data)

    return clustering


# Assigns new data points to established clusters
def predict_cluster(data, kpca, clustering):
    projected_data = kpca.transform(data)
    clustering_labels = clustering.predict(projected_data)

    return clustering_labels, projected_data


# Visualizes the results of clustering as 2D or 3D scatter plots
def visualize_clustering(train_clusters, train_labels, train_projected, test_clusters=None, test_labels=None,
                         test_projected=None, fontsize=5):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot()
    ax.set_title("K-Means Clustering of Description Embeddings Projected into 2D Space Using KPCA",
                 fontsize=fontsize + 10)

    if test_labels is not None:
        all_projected = np.concatenate((train_projected, test_projected), axis=0)
        all_clusters = np.concatenate((train_clusters, test_clusters), axis=0)
        all_labels = train_labels + test_labels
        for i, txt in enumerate(all_labels):
            ax.scatter(all_projected[i, 0], all_projected[i, 1], color=colors[all_clusters[i]])
            ax.text(all_projected[i, 0], all_projected[i, 1], txt, fontsize=fontsize)
    else:
        for i, txt in enumerate(train_labels):
            ax.scatter(train_projected[i, 0], train_projected[i, 1], color=colors[train_clusters[i]])
            ax.text(train_projected[i, 0], train_projected[i, 1], txt, fontsize=fontsize)

    if test_labels is None:
        all_clusters = train_clusters
        all_projected = train_projected

    # draw shapes around clusters
    for cluster in np.unique(all_clusters):
        points = all_projected[all_clusters == cluster]

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
            offset = (np.max(all_projected[:, 1]) - np.min(all_projected[:, 1])) * 0.025
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
            offset = (np.max(all_projected[:, 1]) - np.min(all_projected[:, 1])) * 0.025
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
        offset = (np.max(all_projected[:, 1]) - np.min(all_projected[:, 1])) * 0.025
        ax.text(np.min(points[:, 0]), np.max(points[:, 1]) + offset, "Cluster " + str(cluster),
                fontsize=fontsize + 5, alpha=0.5, color=colors[cluster])

    # plt.savefig("clustering_kpca.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show(block=True)
