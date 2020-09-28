from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as net
import numpy as np
import scipy.sparse as sp
import seaborn as sb
import pandas as pd
from gensim.models import LdaModel
from scipy.stats import entropy
from tqdm import tqdm
from matplotlib.cbook import boxplot_stats


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
    sim = {}
    with Pool(8) as p:
        doc_sim = p.map(partial(document_sim_matrix_par, td_matrix), range(td_matrix.shape[0]))
        for dictionary in tqdm(doc_sim):
            sim.update(dictionary)
    sp.save_npz(sp.dok_matrix(sim), "/Generated Files/adj_matrix")
    return sim


def document_sim_matrix_par(td_matrix, doc_id):
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

# def parallel(index, document):
#     for second_index in range(index):
#         sim = similarity_between_documents(document_topics[index], document_topics[second_index])
#         doc_sim_matrix[index, second_index] = sim
#
#
# def parallel_doc_sim_matrix(document_topics):
#     with Pool(8) as p:
#         p.starmap(parallel, [x for x in enumerate(document_topics)])


def evaluate_doc_topic_distributions(dtm, show=True, tell=True, prune=True):
    sb.set_theme(style="whitegrid")
    # Topic-Doc
    lens = []
    zeros = []
    for i in tqdm(range(0, dtm.shape[1])):
        topic = dtm.getcol(i).nonzero()[0]
        lens.append(len(topic))
        if len(topic) == 0:
            zeros.append(i)
    if tell:
        print("Topic-Doc distributions.")
        print("Minimum: " + str(min(lens)))
        print("Maximum: " + str(max(lens)))
        print("Average: " + str(np.mean(lens)))
        print("Entropy: " + str(entropy(lens, base=len(lens))))
        print("Zeros: " + str(len(zeros)))

    ax = sb.boxplot(x=lens)
    if prune:
        outlier_nums = [y for stat in boxplot_stats(lens) for y in stat['fliers']]
        outliers = [lens.index(x) for x in outlier_nums]
        outliers.extend(zeros)
        outliers = list(set(outliers))
        dtm = slice_sparse_col(dtm, outliers)
    if show:
        plt.show()

    # Doc-Topic
    lens = []
    zeros = []
    for i in tqdm(range(0, dtm.shape[0])):
        topic = dtm.getrow(i).nonzero()[0]
        lens.append(len(topic))
        if len(topic) == 0:
            zeros.append(i)
    if tell:
        print("Doc-Topic distributions.")
        print("Minimum: " + str(min(lens)))
        print("Maximum: " + str(max(lens)))
        print("Average: " + str(np.mean(lens)))
        print("Entropy: " + str(entropy(lens, base=len(lens))))
        print("Zeros: " + str(len(zeros)))
    if prune:
        dtm = sp.csr_matrix(dtm)
        dtm = slice_sparse_row(dtm, zeros)
    if show:
        ax = sb.boxplot(x=lens)
        plt.show()

    return dtm


def slice_sparse_col(M, col):
    col.sort()
    ms = []
    prev = -1
    for c in col:
        ms.append(M[:, prev+1:c-1])
        prev = c
    ms.append(M[:, prev+1:])
    return sp.hstack(ms)


def slice_sparse_row(M, row):
    row.sort()
    ms = []
    prev = -1
    for r in row:
        ms.append(M[prev+1:r-1, :])
        prev = r
    ms.append(M[prev+1:, :])
    return sp.vstack(ms)


if __name__ == '__main__':
    # Loading stuff and initialisation
    topic_doc_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
    topic_doc_matrix = evaluate_doc_topic_distributions(topic_doc_matrix)
    evaluate_doc_topic_distributions(topic_doc_matrix, prune=False, show=False)
    lda_model = LdaModel.load("LDA/model/docu_model")
    document_similarity_matrix_xyz(topic_doc_matrix)
    # documents = preprocess('documents.json')
    # create_document_topics(documents[2])
