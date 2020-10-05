from typing import List, Dict

from gensim.models import LdaModel

from LDA.lda import load_lda
from standard_random_walk import construct_transition_probability_matrix
import numpy as np
import scipy.sparse as sp


def cluster_page_rank(model: LdaModel, adj_matrix):
    adj_matrix = construct_transition_probability_matrix(adj_matrix)
    topic_clusters_plus_words = model.show_topics(lda.num_topics, formatted=False)

    sen_scores = document_scores(adj_matrix, [], topic_clusters_plus_words)
    return sen_scores


def document_scores(adj_matrix, documents: List[str], topic_clusters, dmp_factor=0.85, num_steps=10):
    e = np.ones(adj_matrix.shape[0])
    document_score = np.ones(len(documents))

    m_star = construct_new_transition_matrix(adj_matrix, documents, topic_clusters)

    for x in range(num_steps):
        document_score = dmp_factor * np.dot(document_score, m_star.T) + (1 - dmp_factor) / len(documents) * e
    return document_score


def construct_new_transition_matrix(adj_matrix,
                                    documents: List[str],
                                    clusters: Dict[int, List[str]],
                                    dmp_factor=0.85) -> np.array:
    """
    This function construct the new transition matrix based on the cluster paper
    It calculates the similarity between
    Similarity(document -> document), Similarity(document -> cluster), Similarity(cluster -> document set)
    and combines into a matrix
    :param adj_matrix: adj matrix
    :param documents: the documents
    :param clusters: the cluster list
    :param dmp_factor: 0.85
    :return: a transition matrix based on the cluster paper
    """
    # variable initialization
    similarity_doc_clus = np.zeros(shape=(len(documents), len(clusters)))  # Document to Cluster
    similarity_clus_doc_set = np.zeros(shape=(len(clusters),))  # Cluster to Document Set

    # Calculate the similarity scores
    for index, document in enumerate(documents):
        similarity_doc_clus[index] = similarity_between_document_and_clusters(document, clusters)
    for x in clusters:
        similarity_clus_doc_set = similarity_between_cluster_and_document_set(clusters, documents)

    # Normalize similarity vectors
    similarity_clus_doc_set = similarity_clus_doc_set / similarity_clus_doc_set.sum(0)
    similarity_doc_clus = similarity_doc_clus / similarity_doc_clus.sum(0)

    # Creating the new row normalized adjacency matrix
    m_star = dmp_factor * np.dot(similarity_doc_clus, similarity_clus_doc_set) * adj_matrix + (1 - dmp_factor)
    return m_star / m_star.sum(0)


def similarity_between_document_and_clusters(sentence: str, cluster: dict) -> np.array:
    return NotImplemented


def similarity_between_cluster_and_document_set(cluster: dict, document: net.Graph) -> np.array:
    return NotImplemented


if __name__ == '__main__':
    lda = load_lda("LDA/model/docu_model")
    construct_transition_probability_matrix(sp.load_npz("Generated Files/full_matrix.npz")[:1000, :1000])
