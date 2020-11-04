from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

import utility
import query_handling

count_vectorizer = sp.load_npz("Generated Files/count_vec_matrix.npz")

doc2word = utility.load_vector_file("Generated Files/doc2word.csv")
word2vec = utility.load_vector_file("Generated Files/word2vec.csv")
inverse_w2v = {v: k for k, v in word2vec.items()}
dirichlet_prior = sum([len(i) for i in list(doc2word.values())]) / len(doc2word)


def language_model(query: List[str], document_index: int):
    p_wd = []
    document = doc2word[document_index]

    for word in query:
        word_index = inverse_w2v[word]
        N_d = len(document)
        tf = count_vectorizer[document_index, word_index]
        w_freq_in_D = np.sum(count_vectorizer[:, word_index])
        number_of_word_tokens = len(word2vec)
        p_wd.append(
            (N_d / (N_d + dirichlet_prior)) *
            (tf / N_d) +
            (1 - (N_d / (N_d + dirichlet_prior))) *
            (w_freq_in_D / number_of_word_tokens))
    return np.prod(p_wd)


if __name__ == '__main__':
    queries = query_handling.generate_queries(count_vectorizer, word2vec, 10)
    query_words = list(queries.items())[0][1].split(' ')
    query_index = list(queries.items())[0][0]

    print(f"query: {query_words}")
    lst = {}
    with Pool(processes=8) as p:
        max_ = count_vectorizer.shape[0]
        with tqdm(total=max_) as pbar:
            for i, score in enumerate(p.imap(partial(language_model, query_words), range(0, max_))):
                lst[i] = score
                pbar.update()

    sorted_list = list(dict(sorted(lst.items(), key=lambda x: x[1], reverse=True)).keys())
    print(f"index of document: {sorted_list.index(query_index)}")
    print(f"query: {query_words}")
    print(f"number 1: {doc2word[sorted_list[0]]}\n")
    print(f"number 2: {doc2word[sorted_list[1]]}\n")
    print(f"number 3: {doc2word[sorted_list[2]]}\n")
    print(f"real document: {doc2word[query_index]}")
