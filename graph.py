import math
from typing import Dict
import numpy as np
import networkx as net
from gensim.models import LdaModel

from LDA.lda import get_document_topics_from_model
from preprocessing import load_document_file


def similarity_between_documents(d1: Dict[int, float], d2: Dict[int, float], num_of_topics: int):
    similarity = 0
    for number in range(num_of_topics):
        if number in d1 and number in d2:
            similarity += min(d1[number], d2[number])
    return similarity


def construct_transistion_probability_matrix(graph: net.Graph) -> np.ndarray:
    """
    This function constructs a transistion probability matrix
    based on the adjacency matrix of the graph
    :param graph: the sentence graph
    :return: np.array
    """
    adj_matrix = net.adj_matrix(graph).todense()
    row_normalized_matrix = adj_matrix / adj_matrix.sum(axis=0)
    return np.array(row_normalized_matrix, dtype=np.float64)


def document_graph(path_to_file: str) -> net.graph:
    documents = list(load_document_file(path_to_file).values())[:100]
    lda_model = LdaModel.load("LDA/model/docu_model")

    documents_topics = [get_document_topics_from_model(lda_model, x) for x in documents]

    document_graph = net.Graph()

    # iterating over sentences and adding them as nodes
    # comparing all sentences to each other.
    for document in documents:
        document_graph.add_node(document)
        for second_document in documents:
            similarity = similarity_between_documents(
                get_document_topics_from_model(lda_model, document),
                get_document_topics_from_model(lda_model, second_document),
                lda_model.num_topics)
            document_graph.add_edge(document, second_document, weight=similarity)
    return document_graph


if __name__ == '__main__':
    graph = document_graph("documents.json")
