import os
import re
from functools import partial
from multiprocessing import Pool

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


def document_similarity(td_matrix, doc_id):
    """
    Takes a term-document matrix and calculates the similarity for a given id
    against all other id's which are similar to the given document.
    :param td_matrix: term-document matri
    :param doc_id: the id of the given document
    :return: a dictionary of similarity scores
    """
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
    """
    Takes the term-document matrix and splits the matrix into chunks and
    calls the document similarity function on each
    :param td_matrix:
    :param chunk_size:
    :return:
    """
    max = int(td_matrix.shape[0] / chunk_size)
    for i in range(max, max + 1):
        print(f"Starting chunk {i}.")
        start = i * chunk_size
        end = min((i + 1) * chunk_size, td_matrix.shape[0])
        save_document_similarity(td_matrix, start, end)
    print("Done.")


def save_document_similarity(td_matrix, start, end):
    """
    Constructs similarity based on a start and an end index
    :param td_matrix: term-document matrix
    :param start: start index
    :param end: end index
    :return: saves matrix to file
    """
    sim = {}
    with Pool(8) as p:
        similarities = p.map(partial(document_similarity, td_matrix), range(start, end))
    for dictionary in tqdm(similarities):
        sim.update(dictionary)
    dok = sp.dok_matrix((end - start, td_matrix.shape[0]))
    for (a, b), v in sim.items():
        dok[a - start, b] = v
    sp.save_npz(f"Generated Files/adj/adj_matrix{start}-{end}", sp.csr_matrix(dok))
    del sim
    del dok
    del similarities


def stack_matrixes_in_folder(path):
    """
    Stack adj_matrixes on top of each other and return the full adj matrix
    :param path: folder
    :return:
    """
    files = os.listdir(path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    doc = sp.load_npz(path + files[0])
    for file in files[1:]:
        doc2 = sp.load_npz(path + file)
        doc = sp.vstack([doc, doc2])
        print(doc.shape)
    return doc


if __name__ == '__main__':
    # Loading stuff and initialisation
    matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
    doc_sim_chunker(matrix, 500)

    # Save full matrix
    sp.save_npz("Generated Files/full_matrix", stack_matrixes_in_folder("Generated Files/adj/"))

    # Load full matrix
    adj_matrix = sp.load_npz("Generated Files/full_matrix.npz")
