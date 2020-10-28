import numpy as np
import scipy.sparse as sp

from LDA.lda import load_lda
from standard_random_walk import construct_transition_probability_matrix


def cluster_page_rank(adj_matrix: sp.csr_matrix, query: np.ndarray = None) -> np.array:
    adj_matrix = construct_transition_probability_matrix(adj_matrix)
    ranking_of_documents = document_scores(adj_matrix, query=query)
    return ranking_of_documents


def document_scores(adj_matrix, dmp_factor=0.85, num_steps=10, query: np.ndarray = None):
    """
    Calculates the document scores for each document in the adj_matrix
    by constructing the new transition probability matrix and random walking it
    :param adj_matrix: adj_matrix
    :param dmp_factor: 0.85
    :param num_steps: number of power iterations
    :param query: the query to be searched for
    :return: sorted list of document based on their scores
    """
    document_score = np.ones(adj_matrix.shape[0])

    m_star = construct_new_transition_matrix(adj_matrix)

    for x in range(num_steps):
        document_score = dmp_factor * np.dot(document_score, m_star.T) + (1 - dmp_factor) / adj_matrix.shape[0] * query
    return document_score.argsort()[:][::-1]


def construct_new_transition_matrix(adj_matrix, dmp_factor=0.85) -> np.array:
    """
    This function construct the new transition matrix based on the cluster paper
    It calculates the similarity between
    Similarity(document -> document), Similarity(document -> cluster), Similarity(cluster -> document set)
    and combines into a matrix
    :param adj_matrix: adj matrix
    :param dmp_factor: 0.85
    :return: a transition matrix based on the cluster paper
    """
    # Document to Cluster
    doc_clus_sim = np.array(sp.load_npz("Generated Files/topic_doc_matrix.npz")[:500].todense())

    # Cluster to Document Set
    clus_doc_set_sim = np.array(similarity_between_cluster_and_document_set(doc_clus_sim))

    # Creating the new row normalized adjacency matrix
    adj_matrix = dmp_factor * np.dot(doc_clus_sim, clus_doc_set_sim).T * np.array(adj_matrix) + (1 - dmp_factor) / adj_matrix.shape[0]

    # Returning the normalized version
    return adj_matrix


def similarity_between_cluster_and_document_set(td_matrix) -> np.array:
    """
    Calculates the similarity between each cluster and the document set
    :param td_matrix: topic-document matrix
    :return: numpy array
    """
    return td_matrix.sum(axis=0) / td_matrix.shape[0]


if __name__ == '__main__':
    lda = load_lda("LDA/model/docu_model")
    matrix = sp.load_npz("Generated Files/full_matrix.npz")[:1000, :1000]
    ranks = cluster_page_rank(matrix)
    print(ranks)
