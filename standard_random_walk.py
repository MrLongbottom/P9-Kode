import random
from typing import Dict

import numpy as np
import scipy.sparse as sp


def construct_transition_probability_matrix(adj_matrix):
    """
    This function constructs a transition probability matrix
    based on the adjacency matrix of the graph.
    The diagonal is set to 0 in order to avoid self transitions
    :param adj_matrix: an adjacency matrix based on the documents
    :return: np.array
    """
    adj_matrix = adj_matrix + adj_matrix.T
    np.fill_diagonal(adj_matrix, 0)
    row_normalized_matrix = adj_matrix / adj_matrix.sum(axis=0)

    return row_normalized_matrix


def step_vector(adj_matrix) -> np.array:
    """
    This function create a vector with length of the graph
    where one element is 1 and the rest is 0
    :param adj_matrix: adjacency matrix
    :return: np.array
    """
    random_index = random.randrange(0, adj_matrix.shape[0])
    step = np.zeros(adj_matrix.shape[0])
    step[0] = 1
    return step


def random_walk(steps: int, adj_matrix) -> Dict[str, float]:
    """
    Computes the random walk on a sentence graph
    :param adj_matrix: the adj matrix
    :param steps: The number power iterations
    :return: a dict comprised of sentences and their score
    """
    step = step_vector(adj_matrix)
    adj_matrix = adj_matrix / adj_matrix.sum(0)
    for index in range(steps):
        step = step.dot(adj_matrix.T)
    return step


def random_walk_with_teleport(adj_matrix: np.ndarray,
                              teleport_vector: np.ndarray,
                              steps: int = 10,
                              damp_factor=0.85) -> Dict[str, float]:
    """
    This function is just like random walk but with a teleport vector
    :param adj_matrix: the adjecancy
    :param steps: The number power iterations
    :param teleport_vector: a numpy array which indicates which nodes to favor
    :param damp_factor: 0.85
    :return: a dict comprised of sentences and their score
    """
    # normalized_teleport = teleport_vector / teleport_vector.sum(axis=0)
    trans_prob_matrix = construct_transition_probability_matrix(adj_matrix)

    # random start node
    adj_length = adj_matrix.shape[0]
    index = random.randrange(0, adj_length)
    step = np.zeros(adj_length)
    step[index] = 1
    for index in range(steps):
        step = damp_factor * np.dot(step, trans_prob_matrix.T) + (1 - damp_factor) * teleport_vector
    return step


if __name__ == '__main__':
    matrix = construct_transition_probability_matrix(sp.load_npz("new_matrix.npz")[:500, :500])
    list_of_index = random_walk(10, matrix)[:10]
    print(list_of_index)
