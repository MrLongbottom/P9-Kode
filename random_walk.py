from typing import Dict

import networkx as net
import numpy as np
import random
import scipy.sparse as sp

from preprocessing import load_document_file


def construct_transistion_probability_matrix(adj_matrix) -> np.ndarray:
    """
    This function constructs a transistion probability matrix
    based on the adjacency matrix of the graph
    :param graph: the sentence graph
    :return: np.array
    """
    adj_matrix = adj_matrix.todense()
    row_normalized_matrix = adj_matrix / adj_matrix.sum(axis=0)
    return np.array(row_normalized_matrix, dtype=np.float64)


def step_vector(adj_matrix) -> np.array:
    """
    This function create a vector with length of the graph
    where one element is 1 and the rest is 0
    :param graph: a sentence graph
    :return: np.array
    """
    random_index = random.randrange(0, adj_matrix.shape[0])
    step = np.zeros(adj_matrix.shape[0])
    step[random_index] = 1
    return step


def random_walk(steps: int, graph: net.Graph) -> Dict[str, float]:
    """
    Computes the random walk on a sentence graph
    :param steps: The number power iterations
    :param graph: the sentence graph
    :return: a dict comprised of sentences and their score
    """
    trans_prob_matrix = construct_transistion_probability_matrix(sp.load_npz("../P9-Kode/Generated Files/doc_sim_matrix.npz"))
    step = step_vector(trans_prob_matrix)
    for index in range(steps):
        step = np.dot(step, trans_prob_matrix.T)
    return step.argsort()[:][::-1]


if __name__ == '__main__':
    matrix = sp.load_npz("Generated Files/doc_sim_matrix.npz")
    list_of_index = random_walk(10, matrix)
