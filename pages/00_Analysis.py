import streamlit as st
import numpy as np
from code.visualisation import compute_tsne, compute_kpca, \
    plot_2d, plot_3d, plot_map, load_economic_df, add_similarity_heatmap, add_clusters
from code.data_util import get_embeddings
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")


def close_expander():
    st.session_state.expand_input = False
    st.session_state.submit = True


@st.cache_data
def compute_clustering(test_features, embed_type, feature_list, dim_reduct_method, num_clusters, num_dims):
    embed_type = "max" if embed_type == "Maximum of Features" else \
        ("mean" if embed_type == "Mean of Features" else embed_type)
    economic_df = load_economic_df()
    input_features = []
    if "Description" in feature_list:
        embeddings = get_embeddings(embed_type,
                                    test_feature=test_features["Description"])
        input_features += list(np.moveaxis(np.array(embeddings), 0, -1))
        feature_list.remove("Description")
    for feature in feature_list:
        embeddings = get_embeddings(embed_type,
                                    test_feature=test_features[feature],
                                    data_to_embed=list(economic_df[feature]))
        input_features += list(np.moveaxis(np.array(embeddings), 0, -1))
    input_features = np.moveaxis(np.array(input_features), 0, -1)
    if dim_reduct_method == "t-SNE":
        clustered, projected = compute_tsne(num_clusters, num_dims, input_features)
    elif dim_reduct_method == "Kernel PCA":
        clustered, projected = compute_kpca(num_clusters, num_dims, input_features)
    else:
        clustered = None
        projected = None
    return clustered, projected


if "expand_input" not in st.session_state:
    st.session_state.expand_input = True
if "submit" not in st.session_state:
    st.session_state.submit = False

with st.form("Search"):
    with st.expander("Company details", expanded=st.session_state.expand_input):
        text, other = st.columns([1, 1])
        test_df = {}
        with text:
            test_label = st.text_input("Enter company name:",
                                       placeholder="Your company")
            test_df["Description"] = st.text_area("Enter your company description:",
                                                  placeholder="Your company description",
                                                  height=305)
        with other:
            eco_cols = load_economic_df()
            test_df["Industry"] = st.text_input("Industry branch",
                                                placeholder="E.g. Consulting, Manufacturing, Steel production")
            test_df["Products"] = st.text_input("Product",
                                                placeholder="E.g. Food, Wheels, Steel")
            test_df["Customer Base"] = st.text_input("Targeted customer base",
                                                     placeholder="E.g. Consumers, Families, Other companies")
            test_df["Market Positioning"] = st.text_input("Current Market Position",
                                                          placeholder="E.g. Start-Up, Local business, Global leader")
            test_df["Revenue"] = st.text_input("Current Revenue",
                                               placeholder="E.g. 0€, 1 million $")
    with st.expander("Parameters for Clustering", expanded=st.session_state.expand_input):
        params, feat_select = st.columns([1, 1])
        with params:
            method = st.radio("Select dimensionality reduction method", options=["t-SNE", "Kernel PCA"])
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=24, value=10)
            dimensions = st.radio("Select number of dimensions to use for clustering", options=[2, 3], horizontal=True)
        with feat_select:
            embedding_type = st.radio("Select embedding",
                                      options=["Maximum of Features", "Mean of Features", "Concatenated Features"],
                                      help="Select how text data will be embedded. "
                                           "Maximum of Features: Takes the maximum of feature embeddings "
                                           "across sentences in the description.\n"
                                           "Mean of Features: Takes the mean of feature embeddings "
                                           "across sentences in the description.\n"
                                           "Concatenated Features: Concatenates all features into a "
                                           "large input feature."
                                           "Computation usually takes longer using this method."
                                      )
            features = st.multiselect("Select features", options=["Description", "Industry", "Products",
                                                                  "Customer Base", "Market Positioning", "Revenue"],
                                      default=["Description"])
    st.form_submit_button("Submit", on_click=close_expander, use_container_width=True, type="primary")
if st.session_state.submit:
    test_label = "Your company" if test_label is None else test_label
    clusters, projected_coords = compute_clustering(test_features=test_df, feature_list=features,
                                                    embed_type=embedding_type,
                                                    dim_reduct_method=method,
                                                    num_clusters=n_clusters, num_dims=dimensions)
    with st.expander("Competitors", expanded=True):
        df = load_economic_df()
        competitors = df[clusters[:-1] == clusters[-1]]
        if not competitors.empty:
            st.write(f"Your competitors are:")
            st.dataframe(competitors)
            st.write(
                f"Your competitors are working in the following industries: "
                f"{np.unique(competitors['Industry'].values)}")
            if dimensions == 2:
                figure = plot_2d(clusters[:-1], projected_coords[:-1])
                figure = add_clusters(figure, clusters[:-1], projected_coords[:-1])
                figure.add_trace(go.Scatter(
                    x=[projected_coords[-1, 0]],
                    y=[projected_coords[-1, 1]],
                    marker=dict(
                        color=px.colors.qualitative.Light24[clusters[-1]],
                        size=20,
                        line=dict(
                            color='White',
                            width=2
                        ), ),
                    mode="markers+text",
                    text=test_label,
                    hoverinfo="none",
                    textposition="top center",
                    showlegend=False
                ))
                figure.update_layout(height=400)
            else:
                figure = plot_3d(clusters[:-1], projected_coords[:-1])
                figure.update_traces(marker=dict(opacity=0.3))
                figure.add_trace(go.Scatter3d(
                    x=[projected_coords[-1, 0]],
                    y=[projected_coords[-1, 1]],
                    z=[projected_coords[-1, 2]],
                    marker=dict(
                        color=px.colors.qualitative.Light24[clusters[-1]],
                        size=15, ),
                    mode="markers+text",
                    hoverinfo="none",
                    text=test_label,
                    textposition="top center",
                    showlegend=False
                ))
                figure.update_layout(height=400)
            plot, map_plot = st.columns([1, 1])
            with plot:
                st.plotly_chart(figure, use_container_width=True)
            with map_plot:
                heatmap = st.checkbox("Overlay similarity heatmap", value=True)
                selected = [cluster if cluster == clusters[-1] else None for cluster in clusters[:-1]]
                map_fig = plot_map(selected, plot_connections=True)
                if heatmap:
                    distances = []
                    point_of_interest = projected_coords[-1]
                    for point in projected_coords[:-1]:
                        distances.append(np.sqrt(
                            np.sum(
                                [np.power((point_2 - point_1), 2)
                                 for point_1, point_2 in zip(point, point_of_interest)])
                        ))
                    min_points = np.min(projected_coords, axis=1)
                    max_points = np.max(projected_coords, axis=1)
                    spread = np.sqrt(np.sum(
                        [np.power((point_2 - point_1), 2)
                         for point_1, point_2 in zip(min_points, max_points)])
                    )
                    max_distance = spread / 5
                    map_fig = add_similarity_heatmap(map_fig, distances, max_distance, clusters[-1], clusters[:-1])
                map_fig.update_layout(height=390)
                st.plotly_chart(map_fig)
        else:
            st.warning("Our clustering predicts that no companies in Saarland are very similar to yours based on your"
                       " given description."
                       " Try to lower the amount of clusters to see which cluster is the most similar.")
