"""
This script contains the functions needed to transform a company's description into low-dimensional embedding.
"""
import numpy as np
import re
from sentence_transformers import SentenceTransformer


# Computes embeddings for descriptions by transforming each sentence in a description using a sentence transformer model
# and then taking the average or maximum across all dimensions.
def get_description_embeddings(descriptions, embed_type="max", **kwargs):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    description_embeddings = []
    if embed_type in ["max", "mean"]:
        maximum = embed_type == "max"
        for description in descriptions:
            sentences = re.split('[.!?]', description)
            sentence_embeddings = []
            for sentence in sentences:
                sentence_embedding = model.encode(sentence)
                sentence_embeddings.append(sentence_embedding)
            if maximum:
                description_embedding = np.max(sentence_embeddings, axis=0)
            else:
                description_embedding = np.mean(sentence_embeddings, axis=0)
            description_embeddings.append(description_embedding)
        return description_embeddings
    else:
        for description in descriptions:
            sentences = re.split('[.!?]', description)
            sentence_embeddings = []
            for sentence in sentences:
                sentence_embedding = model.encode(sentence)
                sentence_embeddings.append(sentence_embedding)
            counter = 0
            while len(sentence_embeddings) < 15:
                sentence_embeddings.append(sentence_embeddings[counter])
                counter += 1
            flattened_embeddings = []
            for embedding in sentence_embeddings:
                flattened_embeddings += list(embedding)
            description_embeddings.append(flattened_embeddings)
        return description_embeddings
