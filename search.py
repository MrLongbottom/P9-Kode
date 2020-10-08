from typing import Dict

import numpy as np

from LDA.lda import load_lda, get_document_topics_from_model
from preprocessing import preprocess_query


def make_personalization_vector(topics: Dict[int, float], num_topics: int):
    vector = np.zeros(num_topics)
    for key, value in topics.items():
        vector[key] = value
    return vector


def search(query: str, model_path: str):
    processed_query = preprocess_query(query)
    lda = load_lda(model_path)
    query_topics = get_document_topics_from_model(processed_query, lda)
    return make_personalization_vector(query_topics, lda.num_topics)


if __name__ == '__main__':
    query = "fodbold spiller"
    search_vector = search(query, "LDA/model/search_model")
    print(search_vector)
