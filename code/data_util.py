import pandas as pd
import numpy as np
import streamlit as st
import paths
import requests
import urllib
import os
from code.description_data import train_labels, train_descriptions
from code.embeddings import get_description_embeddings


def get_economic_data_for_company(company_name):
    economic_dataframe = pd.read_csv('../data/economic_data.csv')
    description_dataframe = pd.read_csv('../data/train.csv')
    company_data = economic_dataframe[economic_dataframe['Company'] == company_name]
    company_description = description_dataframe[description_dataframe['label'] == company_name]['description'].values[0]

    return company_data, company_description


@st.cache_data
def load_economic_df():
    df = pd.read_csv(paths.ECO_CSV)
    return df


def get_lat_lon_of_request(search_string):
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(search_string) + '?format=json'
    response = requests.get(url).json()
    return response[0]["lat"], response[0]["lon"]


@st.cache_data
def load_map_df():
    if os.path.exists(paths.MAP_CSV):
        return pd.read_csv(paths.MAP_CSV)
    data = np.zeros((len(train_labels), 2))
    for i, company in enumerate(train_labels):
        print(i, company)
        try:
            lat, lon = get_lat_lon_of_request(company + ", Saarland")
        except Exception as e:
            print(e)
            lat, lon = None, None
        data[i, 0] = lat
        data[i, 1] = lon
    coords_df = pd.DataFrame(data, columns=["latitude", "longitude"])
    df = pd.merge(pd.DataFrame(train_labels, columns=["labels"]), coords_df, right_index=True, left_index=True)
    df.to_csv(paths.MAP_CSV)
    return df


@st.cache_data
def get_embeddings(embedding_type="max", test_feature=None, data_to_embed=None):
    """Embeds the data given or if none given embeds training descriptions."""
    if data_to_embed is None:
        if test_feature:
            return get_description_embeddings(train_descriptions + [test_feature], embed_type=embedding_type)
        else:
            return get_description_embeddings(train_descriptions, embed_type=embedding_type)
    else:
        if test_feature:
            return get_description_embeddings(data_to_embed + [test_feature], embed_type=embedding_type)
        else:
            return get_description_embeddings(data_to_embed, embed_type=embedding_type)
