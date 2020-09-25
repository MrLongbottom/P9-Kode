import matplotlib.pyplot as plt
import networkx as net
import numpy as np
import scipy.sparse as sp
import seaborn as sb
from gensim.models import LdaModel
from scipy.stats import entropy
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
    sim = {}
    for doc_id in tqdm(range(td_matrix.shape[0])):
        doc = td_matrix.getrow(doc_id)
        topics_in_doc = doc.nonzero()[1]
        for topic_id in topics_in_doc:
            topic = td_matrix.getcol(topic_id)
            docs_in_topic = topic.nonzero()[0]
            Y = {x: topic[x].data[0] for x in docs_in_topic if x != doc_id}
            #filter docs that have already been done, .ie if sim(x,y) check if sim(y,x) already is in sim.
            x = topic[doc_id].data[0]
            test = sum_similarity(x, Y)
            for key, val in test.items():
                sim[doc_id, key] = sim.get((doc_id, key), 0) + val
    return sim

def sum_similarity (x, Y):
    dict = {}
    for id, y in Y.items():
        dict[id] = min(x, y)
    return dict

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
    topic_doc_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
    lda_model = LdaModel.load("LDA/model/docu_model")
    document_similarity_matrix_xyz(topic_doc_matrix)
    # documents = preprocess('documents.json')
    # create_document_topics(documents[2])
