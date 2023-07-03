from code.clustering_tsne import project_tsne, predict_clusters
from code.clustering_kpca import predict_cluster, fit_kernel_pca, cluster_data
from code.description_data import train_labels
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial import ConvexHull
from scipy import interpolate
from code.data_util import load_map_df, load_economic_df


@st.cache_data
def compute_hull(points):
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
        return points[:, 0], points[:, 1]

    # compute convex hull
    try:
        hull = ConvexHull(points)  # computation might not work if data points in a cluster have a very weird position
    except Exception as e:
        print(e)
        return points[:, 0], points[:, 1]
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
    return interp_x, interp_y


@st.cache_data
def compute_tsne(n_clusters, dimensions, train_embeddings):
    # Compute the clustering of the training data
    projected_train_embeddings = project_tsne(data=train_embeddings, dimensions=dimensions,
                                              perplexity=np.sqrt(len(train_embeddings)),
                                              random_state=42)
    clusters, projected = predict_clusters(projected_data=projected_train_embeddings, n_clusters=n_clusters)
    return clusters, projected


@st.cache_data
def compute_kpca(n_clusters, dimensions, train_embeddings):
    kpca = fit_kernel_pca(train_embeddings, n_components=dimensions)
    clustering = cluster_data(train_embeddings, kpca, n_clusters)
    return predict_cluster(train_embeddings, kpca, clustering)


@st.cache_data
def plot_map(clusters, plot_connections=False):
    map_df = load_map_df()
    hover_df = load_economic_df()
    df = pd.merge(map_df, hover_df, right_index=True, left_index=True)
    df["Cluster"] = [str(cluster) if cluster is not None else None for cluster in clusters]
    selected = df.dropna()
    fig = px.scatter_mapbox(selected, lat="latitude", lon="longitude", color="Cluster", custom_data=selected,
                            text="Company", zoom=9,
                            category_orders={"Cluster": list(sorted(selected["Cluster"], key=lambda x: int(x)))},
                            hover_name="Company", hover_data=["Industry", "Customer Base", "Market Positioning"],
                            color_discrete_sequence=px.colors.qualitative.Light24)
    fig.update_layout(mapbox_style='carto-darkmatter')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(mode="markers+text", showlegend=not plot_connections, textposition='top center',
                      hovertemplate='<b>%{customdata[5]}</b><br>'
                                    'Industry: %{customdata[6]} <br>'
                                    'Products: %{customdata[7]} <br>'
                                    'Customer base: %{customdata[8]} <br>'
                                    'Market Position: %{customdata[9]} <br>'
                                    'Revenue: %{customdata[10]}€')
    fig.update_layout(
        height=750,
        hoverlabel=dict(
            font_size=16
        ),
        margin=go.Margin(
            l=0,
            r=0,
            b=0,
            t=10
        )
    )
    if plot_connections:
        data = fig.data
        for i in reversed(range(len(clusters))):
            if clusters[i] is None:
                del clusters[i]
        for cluster in np.unique(clusters):
            lats = selected[selected["Cluster"] == str(cluster)]["latitude"]
            lons = selected[selected["Cluster"] == str(cluster)]["longitude"]
            mean_lat = np.mean(lats)
            mean_lon = np.mean(lons)
            new_lats = []
            new_lons = []
            for lat, lon in zip(lats, lons):
                new_lats += [mean_lat, lat]
                new_lons += [mean_lon, lon]
            fig.add_trace(go.Scattermapbox(
                lat=new_lats,
                lon=new_lons,
                fillcolor=px.colors.qualitative.Light24[cluster],
                mode="lines",
                opacity=0.1,
                line=dict(color=px.colors.qualitative.Light24[cluster]),
                showlegend=False
            ))
        fig.add_traces(data=data)
        if len(np.unique(clusters)) == 1:
            fig.update_traces(marker=dict(color=px.colors.qualitative.Light24[clusters[0]]))
    return fig


@st.cache_data
def add_similarity_heatmap(fig, distances, max_distance, cluster, clusters):
    map_df = load_map_df()
    hover_df = load_economic_df()
    df = pd.merge(map_df, hover_df, right_index=True, left_index=True)
    # df["Cluster"] = [str(cluster) if cluster is not None else None for cluster in clusters]
    df["Distances"] = [max(((max_distance - distance) / max_distance), 0) for distance in distances]
    selected = df.dropna()
    map_fig = px.density_mapbox(selected, lat="latitude", lon="longitude", z="Distances", custom_data=selected,
                                radius=50, opacity=0.5,
                                mapbox_style="carto-darkmatter",
                                color_continuous_scale=px.colors.sequential.Hot,
                                range_color=[0.0, 1.0],
                                labels={"Distances": "Similarity"})
    map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(customdata=selected[clusters == cluster])
    fig.update_traces(hovertemplate='<b>%{customdata[5]}</b><br>'
                                    'Similarity: %{customdata[11]:.2%}<br>'
                                    'Industry: %{customdata[6]} <br>'
                                    'Products: %{customdata[7]} <br>'
                                    'Customer base: %{customdata[8]} <br>'
                                    'Market Position: %{customdata[9]} <br>'
                                    'Revenue: %{customdata[10]}€')
    map_fig.add_traces(data=fig.data)
    map_fig.update_traces(hovertemplate='<b>%{customdata[5]}</b><br>'
                                        'Similarity: %{customdata[11]:.2%}<br>'
                                        'Industry: %{customdata[6]} <br>'
                                        'Products: %{customdata[7]} <br>'
                                        'Customer base: %{customdata[8]} <br>'
                                        'Market Position: %{customdata[9]} <br>'
                                        'Revenue: %{customdata[10]}€')

    return map_fig


@st.cache_data
def plot_2d(clusters, projected):
    str_cluster = [str(cluster) for cluster in clusters]
    df = pd.merge(
        pd.DataFrame(projected, columns=["x", "y"]),
        load_economic_df(),
        right_index=True, left_index=True
    )
    fig = px.scatter(df,
                     x="x",
                     y="y",
                     color=str_cluster,
                     labels={"color": "Cluster"},
                     category_orders={"color": list(sorted(str_cluster, key=lambda x: int(x)))},
                     color_discrete_sequence=px.colors.qualitative.Light24,
                     text="Company",
                     hover_name="Company",
                     custom_data=df,
                     hover_data={
                         "x": False,
                         "y": False,
                         "Industry": True,
                         "Products": True,
                         "Customer Base": True,
                         "Market Positioning": True,
                         "Revenue": True})
    fig.update_traces(textposition='top center',
                      hovertemplate='<b>%{customdata[3]}</b><br>'
                                    'Industry: %{customdata[4]} <br>'
                                    'Products: %{customdata[5]} <br>'
                                    'Customer base: %{customdata[6]} <br>'
                                    'Market Position: %{customdata[7]} <br>'
                                    'Revenue: %{customdata[8]}€')
    fig.update_layout(
        height=750,
        hoverlabel=dict(
            font_size=16
        ),
        margin=go.Margin(
            l=0,
            r=0,
            b=0,
            t=10
        )
    )
    return fig


@st.cache_data
def plot_3d(clusters, projected):
    str_cluster = [str(cluster) for cluster in clusters]
    df = pd.merge(
        pd.DataFrame(projected),
        load_economic_df(),
        right_index=True, left_index=True
    )
    fig = px.scatter_3d(df,
                        x=0,
                        y=1,
                        z=2,
                        color=str_cluster,
                        labels={"color": "Cluster"},
                        custom_data=df,
                        category_orders={"color": list(sorted(str_cluster, key=lambda x: int(x)))},
                        color_discrete_sequence=px.colors.qualitative.Light24,
                        text=train_labels,
                        hover_name="Company",
                        hover_data={
                            "Industry": True,
                            "Products": True,
                            "Customer Base": True,
                            "Market Positioning": True,
                            "Revenue": True})

    fig.update_traces(textposition='top center',
                      hovertemplate='<b>%{customdata[4]}</b><br>'
                                    'Industry: %{customdata[5]} <br>'
                                    'Products: %{customdata[6]} <br>'
                                    'Customer base: %{customdata[7]} <br>'
                                    'Market Position: %{customdata[8]} <br>'
                                    'Revenue: %{customdata[9]}€', )
    fig.update_layout(
        height=750,
        hoverlabel=dict(
            font_size=16
        ),
        margin=go.Margin(
            l=0,
            r=0,
            b=0,
            t=10
        )
    )
    return fig


@st.cache_data
def add_clusters(fig, clusters, projected):
    traces = []
    for cluster in np.unique(clusters):
        points = projected[clusters == cluster]
        cluster_x, cluster_y = compute_hull(points)
        mean_x = np.mean(cluster_x)
        mean_y = np.mean(cluster_y)
        traces.append(go.Scatter(
            x=cluster_x,
            y=cluster_y,
            fill="toself",
            fillcolor=px.colors.qualitative.Light24[cluster],
            opacity=0.1,
            mode="lines",
            line=dict(color=px.colors.qualitative.Light24[cluster]),
            text=f"Cluster {cluster}",
            textposition="middle center",
            showlegend=False,
            hoverinfo="none"
        ))
        traces.append(go.Scatter(
            x=[mean_x],
            y=[mean_y],
            mode="text",
            fillcolor=px.colors.qualitative.Light24[cluster],
            text=f"Cluster {cluster}",
            textposition="middle center" if len(points[:, 0]) > 1 else "bottom right",
            textfont=dict(color=px.colors.qualitative.Light24[cluster]),
            showlegend=False,
            hoverinfo="none"
        ))
    new_fig = go.Figure(data=traces)
    new_fig.add_traces(data=fig.data)
    new_fig.update_layout(
        height=750,
        hoverlabel=dict(
            font_size=16
        ),
        margin=go.Margin(
            l=0,
            r=0,
            b=0,
            t=10
        )
    )
    new_fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=True, mirror=True, showline=True)
    new_fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=True, mirror=True, showline=True)
    return new_fig


@st.cache_data
def add_industry_clusters(fig, projected):
    df = pd.merge(
        pd.DataFrame(projected),
        load_economic_df(),
        right_index=True, left_index=True
    )
    traces = []
    for i, industry in enumerate(np.unique(df["Industry"])):
        points = projected[df["Industry"] == industry]
        cluster_x, cluster_y = compute_hull(points)
        mean_x = np.mean(cluster_x)
        mean_y = np.mean(cluster_y)
        traces.append(go.Scatter(
            x=cluster_x,
            y=cluster_y,
            fill="toself",
            fillcolor=px.colors.qualitative.Light24[i],
            opacity=0.1,
            mode="lines",
            line=dict(color=px.colors.qualitative.Light24[i]),
            text=f"{industry}",
            textposition="middle center",
            showlegend=False,
        ))
        traces.append(go.Scatter(
            x=[mean_x],
            y=[mean_y],
            mode="text",
            fillcolor=px.colors.qualitative.Light24[i],
            text=f"{industry}",
            textposition="middle center" if len(points[:, 0]) > 1 else "bottom right",
            textfont=dict(color=px.colors.qualitative.Light24[i]),
            showlegend=False,
            hoverinfo="none"
        ))
    new_fig = go.Figure(data=traces)
    new_fig.add_traces(data=fig.data)
    new_fig.update_layout(
        height=750,
        hoverlabel=dict(
            font_size=16
        ),
        margin=go.Margin(
            l=0,
            r=0,
            b=0,
            t=10
        )
    )
    return new_fig
