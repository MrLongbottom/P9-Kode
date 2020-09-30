from multiprocessing import Pool

import h5py
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
        [min(topic_doc_matrix.getrow(d1).toarray()[0][number], topic_doc_matrix.getrow(d2).toarray()[0][number]) for
         number in set(topic_doc_matrix.getrow(0).nonzero()[1]) & set(topic_doc_matrix.getrow(1).nonzero()[1])])


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


def document_similarity_matrix_xyz(td_matrix):
    iterable = [(i, x) for i, x in enumerate(td_matrix)]
    with Pool(8) as p:
        p.map(document_sim_matrix_par, iterable)


def document_sim_matrix_par(pair):
    doc_id = pair[0]
    doc = pair[1]
    topics_in_doc = doc.nonzero()[1]
    skips = 0
    sim_dict = {}
    with h5py.File('adj.hdf5', 'a') as file:
        skipable = file['data'][doc_id].any()
    if not skipable:
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
            with h5py.File('adj.hdf5', 'a') as file:
                file['data'][doc_id].put(list(sim_dict.keys()), list(sim_dict.values()))
        print(f"Doc: {doc_id} Skipped: {skips}")


# def parallel(index, document):
#     for second_index in range(index):
#         sim = similarity_between_documents(document_topics[index], document_topics[second_index])
#         doc_sim_matrix[index, second_index] = sim
#
#
# def parallel_doc_sim_matrix(document_topics):
#     with Pool(8) as p:
#         p.starmap(parallel, [x for x in enumerate(document_topics)])


if __name__ == '__main__':
    # Loading stuff and initialisation
    td_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
    adj_matrix = sp.dok_matrix((td_matrix.shape[0], td_matrix.shape[0]))
    with h5py.File('adj.hdf5', 'a') as file:
        if not file['data']:
            file.create_dataset("data", adj_matrix.shape)

    with h5py.File('adj.hdf5', 'a') as file:
        skipable = file['data'].any()
    document_similarity_matrix_xyz(td_matrix)
    # documents = preprocess('documents.json')
    # create_document_topics(documents[2])
