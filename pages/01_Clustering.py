import streamlit as st
import numpy as np
from code.data_util import get_embeddings, load_economic_df, load_map_df
from code.visualisation import compute_tsne, compute_kpca, \
    plot_2d, plot_3d, plot_map, add_clusters


st.set_page_config(layout="wide")


@st.cache_data
def compute_clustering(embed_type, feature_list: list[str], dim_reduct_method, num_clusters):
    embed_type = "max" if embed_type == "Maximum of Features" else \
        ("mean" if embed_type == "Mean of Features" else embed_type)
    economic_df = load_economic_df()
    map_df = load_map_df()
    input_features = []
    if "Description" in feature_list:
        embeddings = get_embeddings(embed_type)
        input_features += list(np.moveaxis(np.array(embeddings), 0, -1))
        feature_list.remove("Description")
    if "Location" in feature_list:
        input_features.append(list(map_df["latitude"]))
        input_features.append(list(map_df["longitude"]))
        feature_list.remove("Location")
    for feature in feature_list:
        embeddings = get_embeddings(embed_type, data_to_embed=list(economic_df[feature]))
        input_features += list(np.moveaxis(np.array(embeddings), 0, -1))
    input_features = np.moveaxis(np.array(input_features), 0, -1)
    if dim_reduct_method == "t-SNE":
        clusters_2d, projected_2d = compute_tsne(num_clusters, 2, input_features)
        clusters_3d, projected_3d = compute_tsne(num_clusters, 3, input_features)
    elif dim_reduct_method == "Kernel PCA":
        clusters_2d, projected_2d = compute_kpca(num_clusters, 2, input_features)
        clusters_3d, projected_3d = compute_kpca(num_clusters, 3, input_features)
    else:
        raise ValueError("Method not supported!")
    return {"2d": (clusters_2d, projected_2d),
            "3d": (clusters_3d, projected_3d)}


selection_area, plot_area = st.columns([1, 4])
with selection_area:
    with st.form("Clustering parameters"):
        embedding_type = st.radio("Select embedding",
                                  options=["Maximum of Features", "Mean of Features", "Concatenated Features"],
                                  help="Select how text data will be embedded. "
                                       "Maximum of Features: Takes the maximum of feature embeddings "
                                       "across sentences in the description.\n"
                                       "Mean of Features: Takes the mean of feature embeddings "
                                       "across sentences in the description.\n"
                                       "Concatenated Features: Concatenates all features into a large input feature. "
                                       "Computation usually takes longer using this method.")
        method = st.radio("Select dimensionality reduction method", options=["t-SNE", "Kernel PCA"])
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=24, value=10)
        features = st.multiselect("Select features", options=["Description", "Industry", "Products",
                                                              "Customer Base", "Market Positioning", "Revenue"],
                                  default=["Description"])
        st.form_submit_button("Compute clustering")
    plot_data_dict = compute_clustering(embedding_type, features, method, n_clusters)
    fig_2d = plot_2d(plot_data_dict["2d"][0], plot_data_dict["2d"][1])
    fig_3d = plot_3d(plot_data_dict["3d"][0], plot_data_dict["3d"][1])
    dimensions = st.radio("Select number of dimensions to display", options=["2d", "3d"])
with plot_area:
    compare_mode = st.radio("Plot Style", options=["Cluster Plot", "Map", "Both"], horizontal=True)
    if dimensions == "2d":
        if compare_mode == "Map":
            new_fig = plot_map(clusters=plot_data_dict["2d"][0])
            st.plotly_chart(new_fig, use_container_width=True)
            st.warning("Some companies might be missing on this map!")
        elif compare_mode == "Cluster Plot":
            new_fig = add_clusters(fig_2d, plot_data_dict["2d"][0], plot_data_dict["2d"][1])
            st.plotly_chart(new_fig, use_container_width=True)
        else:
            plot, map_col = st.columns([1, 1])
            with plot:
                new_fig = add_clusters(fig_2d, plot_data_dict["2d"][0], plot_data_dict["2d"][1])
                st.plotly_chart(new_fig, use_container_width=True)
            with map_col:
                new_fig = plot_map(clusters=plot_data_dict["2d"][0])
                st.plotly_chart(new_fig, use_container_width=True)
                st.warning("Some companies might be missing on this map!")
    else:
        if compare_mode == "Map":
            new_fig = plot_map(clusters=plot_data_dict["3d"][0])
            st.plotly_chart(new_fig, use_container_width=True)
            st.warning("Some companies might be missing on this map!")
        elif compare_mode == "Cluster Plot":
            new_fig = add_clusters(fig_2d, plot_data_dict["2d"][0], plot_data_dict["2d"][1])
            st.plotly_chart(new_fig, use_container_width=True)
        else:
            plot, map_col = st.columns([1, 1])
            with plot:
                st.plotly_chart(fig_3d, use_container_width=True)
            with map_col:
                new_fig = plot_map(clusters=plot_data_dict["3d"][0])
                st.plotly_chart(new_fig, use_container_width=True)
                st.warning("Some companies might be missing on this map!")
        st.plotly_chart(fig_3d, use_container_width=True)


