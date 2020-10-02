from functools import partial
from multiprocessing import Pool

import numpy as np
import networkx as net
import scipy.sparse as sp
from tqdm import tqdm


def similarity_between_documents(d1: int, d2: int):
    """
    Computes similarity between two documents by summing the minimum of their topic distribution.
    :param d1: document 1
    :param d2: document 2
    :return: a similarity value between 0 and 1
    """
    sim_set = set(matrix.getrow(0).nonzero()[1]) & set(matrix.getrow(1).nonzero()[1])
    if len(sim_set) >= 0:
        return sum(
            [min(matrix.getrow(d1).toarray()[0][number], matrix.getrow(d2).toarray()[0][number]) for
             number in sim_set])
    else:
        return 0


def document_similarity_matrix(matrix) -> net.graph:
    """
    Todo still not optimized
    """
    length = matrix.shape[0]
    doc_sim_matrix = sp.dok_matrix((length, length))
    for index in tqdm(range(length)):
        for second_index in range(index):
            sim = similarity_between_documents(index, second_index)
            doc_sim_matrix[index, second_index] = sim
    sp.save_npz("Generated Files/doc_sim_matrix", sp.csr_matrix(np.add(doc_sim_matrix, doc_sim_matrix.transpose())))
    return doc_sim_matrix


def document_similarity(td_matrix, doc_id):
    doc = td_matrix.getrow(doc_id)
    topics_in_doc = doc.nonzero()[1]
    skips = 0
    sim_dict = {}
    for topic_id in topics_in_doc:
        topic = td_matrix.getcol(topic_id)
        docs_in_topic = topic.nonzero()[0]
        len1 = len(docs_in_topic)
        # filter docs that have already been done, .ie documents earlier in the loop
        docs_in_topic = [x for x in docs_in_topic if doc_id <= x]
        len2 = len(docs_in_topic)
        if len1 != len2:
            skips += len1 - len2
        Y = {x: topic[x].data[0] for x in docs_in_topic if doc_id <= x}
        x = topic[doc_id].data[0]
        similarity_sum = {id: min(x, y) for id, y in Y.items()}
        for key, val in similarity_sum.items():
            sim_dict[doc_id, key] = sim_dict.get((doc_id, key), 0) + val
    print(f"Doc: {doc_id} Skipped: {skips}")
    return sim_dict


def doc_sim_chunker(td_matrix, chunk_size):
    max = int(td_matrix.shape[0] / chunk_size)
    for i in range(213, max):
        print(f"Starting chunk {i}.")
        start = i*chunk_size
        end = min((i+1)*chunk_size, td_matrix.shape[0])
        document_similarity_matrix_xyz(td_matrix, start, end)
    print("Done.")


def document_similarity_matrix_xyz(td_matrix, start, end):
    sim = {}
    with Pool(5) as p:
        test = p.map(partial(document_similarity, td_matrix), range(start, end))
    for dictionary in tqdm(test):
        sim.update(dictionary)
    dok = sp.dok_matrix((td_matrix.shape[0], td_matrix.shape[0]))
    for (a, b), v in sim.items():
        dok[a, b] = v
    sp.save_npz(f"Generated Files/adj/adj_matrix{start}-{end}", sp.csr_matrix(dok))
    del sim
    del dok
    del test


def inner_graph_func(index):
    document_graph.add_node(index)
    for second_index in range(index):
        similarity = similarity_between_documents(index, second_index)
        document_graph.add_edge(index, second_index, weight=similarity)


def make_document_graph(matrix: sp.dok_matrix):
    with Pool(8) as p:
        max_ = matrix.shape[0]
        with tqdm(total=max_) as pbar:
            for i, _ in enumerate(p.imap_unordered(inner_graph_func, range(matrix.shape[0]))):
                pbar.update()
    return document_graph


def make_node_graph(matrix):
    node_graph = net.Graph()
    for document in range(matrix.shape[0]):
        node_graph.add_node(document)
    net.write_gpickle(node_graph, "Generated Files/graph")
    return node_graph


def add_similarity_to_node_graph(node_graph: net.Graph):
    with Pool(8) as p:
        max_ = matrix.shape[0]
        with tqdm(total=max_) as pbar:
            for i, _ in enumerate(p.imap_unordered(partial(add_sim_sub_func, node_graph), node_graph.nodes)):
                pbar.update()
    return node_graph


def add_sim_sub_func(node_graph, document):
    for second_document in range(document):
        node_graph.add_edge(document, second_document,
                            weigth=similarity_between_documents(document, second_document))


def load_node_graph(path: str):
    return net.read_gpickle(path)


if __name__ == '__main__':
    # Loading stuff and initialisation
    document_graph = net.Graph()
    matrix = sp.load_npz("Generated Files/test_topic_doc_matrix.npz")
    # node_graph = make_node_graph(matrix)
    # node_graph = add_similarity_to_node_graph(node_graph)
    # net.write_gpickle(node_graph, "Generated Files/graph")
    doc_sim_chunker(matrix, 100)
