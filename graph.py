from functools import partial
from multiprocessing import Pool

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
    return sum(
        [min(matrix.getrow(d1).toarray()[0][number], matrix.getrow(d2).toarray()[0][number]) for
         number in set(matrix.getrow(0).nonzero()[1]) & set(matrix.getrow(1).nonzero()[1])])


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
    sp.save_npz("Generated Files/doc_sim_matrix", doc_sim_matrix)
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


def document_chunk_similarity(td_matrix, chunk):
    doc_sim = {}
    for document_id in chunk:
        doc_sim.update(document_similarity(td_matrix, document_id))
    return doc_sim, chunk[0]


def document_similarity_matrix_xyz(td_matrix, chunks):
    sim = {}
    with Pool(8) as p:
        doc_sim, name = p.map(partial(document_chunk_similarity, td_matrix), chunks)
    for dictionary in tqdm(doc_sim):
        sim.update(dictionary)
    dok = sp.dok_matrix((td_matrix.shape[0], td_matrix.shape[0]))
    for (a, b), v in sim.items():
        dok[a, b] = v
    sp.save_npz(f"Generated Files/adj_matrix{name}", sp.csr_matrix(dok))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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


if __name__ == '__main__':
    # Loading stuff and initialisation
    matrix = sp.load_npz("Generated Files/count_vec_matrix.npz")
    document_graph = net.Graph()
    net.write_gpickle(document_graph, "Generated Files/graph")
    # document_similarity_matrix_xyz(td_matrix)
    # chunks = chunks(range(matrix.shape[0]), 100)
    #
    # document_similarity_matrix_xyz(matrix, chunks)
    make_document_graph(matrix)
