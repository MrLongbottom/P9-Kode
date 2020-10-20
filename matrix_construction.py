import os
import re
from functools import partial
from multiprocessing import Pool
import  numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def doc_sim_chunker(td_matrix: sp.spmatrix, chunk_size: int, pool: int):
    """
    Takes the topic-document matrix and splits the matrix into chunks
    and calls the document similarity function on each.
    This is done to save intermediate progress in files before cleaning memory and continuing.
    :param td_matrix: topic-document distribution matrix in sparse format.
    :param chunk_size: how many documents to run before saving.
    :param pool: how many threads to use.
    """
    max = int(td_matrix.shape[0] / chunk_size)
    for i in range(0, max + 1):
        print(f"Starting chunk {i}.")
        start = i * chunk_size
        end = min((i + 1) * chunk_size, td_matrix.shape[0])
        save_document_similarity(td_matrix, start, end, pool)
    print("Done.")


def save_document_similarity(td_matrix: sp.spmatrix, start: int, end: int, pool: int):
    """
    Constructs similarity based on a start and an end index. Saves matrix to file.
    :param td_matrix: topic-document matrix
    :param start: start index
    :param end: end index
    :param pool: how many threads to use
    """
    with Pool(pool) as p:
        similarities = p.map(partial(document_similarity, td_matrix), range(start, end))
    matrix = sp.vstack(similarities)
    sp.save_npz(f"Generated Files/adj/adj_matrix{start}-{end}", sp.csr_matrix(matrix))
    del matrix
    del similarities


def document_similarity(td_matrix: sp.spmatrix, doc_id: int):
    """
    Takes a topic-document matrix and calculates the similarity for a given id
    against all other id's which are similar to the given document.
    :param td_matrix: topic-document matrix
    :param doc_id: the id of the given document
    :return: a coo_matrix consisting of one documents similarity score (one row).
    """
    doc = td_matrix.getrow(doc_id)
    topics_in_doc = doc.nonzero()[1]
    rows = np.array([])
    cols = np.array([])
    vals = np.array([])
    for topic_id in topics_in_doc:
        topic = td_matrix.getcol(topic_id)
        docs_in_topic = topic.nonzero()[0]
        # filter docs that have already been done, .ie documents earlier in the loop
        docs_in_topic = [d for d in docs_in_topic if doc_id < d]
        # put documents that share the same topic as the main document into a dictionary.
        # mapping doc_id -> topic distribution value
        Y = {y: topic[y].data[0] for y in docs_in_topic if doc_id < y}
        x = topic[doc_id].data[0]
        similarity_sum = similarity_function(x, Y)
        # add values to numpy arrays based on row_ids, column_ids and values
        # this is to efficiently construct a coo_matrix
        rows = np.concatenate((rows, np.zeros(len(similarity_sum))))
        cols = np.concatenate((cols, np.array(list(similarity_sum.keys()))))
        vals = np.concatenate((vals, np.array(list(similarity_sum.values()))))
    # construct similarity coo_matrix based on the documents that share topics with the main document.
    sim_dict = sp.coo_matrix((vals, (rows, cols)), shape=(1, td_matrix.shape[0]))
    print(f"Doc: {doc_id} done.")
    return sim_dict


def similarity_function(x, Y):
    """
    Based on topic-document distributions calculates similarities between main document and other documents.
    :param x: main document's distribution to specific topic
    :param Y: other document's distribution to same topic
    :return: dict where each element y in Y, is the minimum of y and x
    """
    return {id: min(x, y) for id, y in Y.items()}


def stack_matrices_in_folder(path: str):
    """
    Stack adj_matrices on top of each other and return the full adj matrix.
    :param path: folder path
    :return: stacked matrix
    """
    files = os.listdir(path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    doc = sp.load_npz(path + files[0])
    for file in files[1:]:
        doc2 = sp.load_npz(path + file)
        doc = sp.vstack([doc, doc2])
        print(doc.shape)
    return doc


def matrix_connection_check(adj_matrix, node=0, visited=None):
    if node == 0:
        visited = [False for x in range(0, adj_matrix.shape[0])]
    visited[node] = True
    neighboors = [x for x in adj_matrix.getrow(0).nonzero()[1] if visited[x] is False]
    for n in neighboors:
        matrix_connection_check(adj_matrix, n, visited)
    if node == 0:
        if False in visited:
            print(f"Only found {len([x for x in visited if x is True])} connected nodes, out of {adj_matrix.shape[0]}.")
            return False
        else:
            return True


if __name__ == '__main__':
    # Loading topic-document distribution matrix and initialisation
    # whether csr_matrix or csc_matrix is faster will probably depend on the number of topics per document.
    #matrix = sp.csr_matrix(sp.load_npz("Generated Files/topic_doc_matrix.npz"))
    #doc_sim_chunker(matrix, 500, 8)

    # Save full matrix
    #sp.save_npz("Generated Files/full_matrix", stack_matrices_in_folder("Generated Files/adj/"))

    # Load full matrix
    adj_matrix = sp.load_npz("Generated Files/full_matrix.npz")
    matrix_connection_check(adj_matrix)
