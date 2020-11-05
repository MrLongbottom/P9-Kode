import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, List
import preprocessing
import utility
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
import scipy.sparse as sp
import language_model
import numpy as np

count_vectorizer = sp.load_npz("Generated Files/count_vec_matrix.npz")

doc2word = utility.load_vector_file("Generated Files/doc2word.csv")
word2vec = utility.load_vector_file("Generated Files/word2vec.csv")
inverse_w2v = {v: k for k, v in word2vec.items()}
dirichlet_prior = sum([len(i) for i in list(doc2word.values())]) / len(doc2word)

dt_matrix = sp.load_npz("Generated Files/topic_doc_matrix.npz")
tw_matrix = sp.load_npz("Generated Files/topic_word_matrix.npz")


def lda_evaluate_word_doc(word, document_index):
    word_index = inverse_w2v[word]
    word_topics = tw_matrix.getcol(word_index)
    doc_topics = dt_matrix[document_index].T
    score = word_topics.multiply(doc_topics).sum()
    return score


def lda_evaluate_query_doc(query: List[str], document_index: int):
    p_wd = []
    for word in query:
        p_wd.append(lda_evaluate_word_doc(word, document_index))
    return np.prod(p_wd)


def lda_evaluate_query(query_index, query_words, tell=False):
    lst = {}
    """
    with Pool(processes=8) as p:
        max_ = count_vectorizer.shape[0]
        with tqdm(total=max_) as pbar:
            for i, score in enumerate(p.imap(partial(lda_evaluate_query_doc, query_words), range(0, max_))):
                lst[i] = score
                pbar.update()
    """
    for doc_id in tqdm(range(count_vectorizer.shape[0])):
        lst[doc_id] = lda_evaluate_query_doc(query_words, doc_id)

    sorted_list = list(dict(sorted(lst.items(), key=lambda x: x[1], reverse=True)).keys())
    if tell:
        print(f"query: {query_words}")
        print(f"index of document: {sorted_list.index(query_index)}")
        print(f"number 1: {doc2word[sorted_list[0]]}\n")
        print(f"number 2: {doc2word[sorted_list[1]]}\n")
        print(f"number 3: {doc2word[sorted_list[2]]}\n")
        print(f"real document: {doc2word[query_index]}")
    return sorted_list.index(query_index), lst


def generate_queries(count_matrix, words: Dict[int, str], count: int, min_length: int = 1, max_length: int = 4):
    """
    Generates queries for random documents based on tfidf values
    :param count_matrix: CountVectorization matrix
    :param words: words dictionary
    :param count: number of queries wanted
    :param min_length: min words per query (exact length is random)
    :param max_length: max words per query (exact length is random)
    :return: dictionary mapping document ids to queries
    """
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(count_matrix)
    queries = {}
    documents_count = tfidf_matrix.shape[0]
    for i in tqdm(range(count)):
        doc_id = random.randrange(0, documents_count)
        query_length = random.randrange(min_length, max_length+1)
        query = []
        doc_vec = tfidf_matrix.getrow(doc_id)
        word_ids = doc_vec.toarray()[0].argsort()[-query_length:][::-1]
        for word_id in word_ids:
            word = words[word_id]
            query.append(word)
        query = ' '.join(query)
        queries[doc_id] = query
    return queries


def check_valid_queries(queries):
    """
    Checks to make sure that each word in the queries actually exists in the original document.
    :param queries: dict of original documents pointing to generated queries.
    :return: True if valid, otherwise returns data to identify the problem.
    """
    fails = []
    for k, v in queries.items():
        for word in v.split(' '):
            if word not in doc2word[k]:
                print("Not in there!")
                fails.append((k, v, word, doc2word[k]))
    if len(fails) > 0:
        return fails
    else:
        return True


def preprocess_query(query: str, word_check=True):
    # cut off words that are used too often or too little (max/min document frequency) or are stop words
    step = 1
    print(f'Step {step}: stop words and word frequency.')
    words = preprocessing.cut_off_words([query], 1.0, 1)

    print(len(words))
    if word_check:
        # cut off words that are not used in danish word databases or are wrong word type
        step += 1
        print(f"Step {step}: word databases and POS-tagging.")
        words = preprocessing.word_checker(words)

    # Stemming to combine word declensions
    step += 1
    print(f"Step {step}: Apply Stemming / Lemming")
    # TODO not sure if just leaving document empty is fine.
    corpus, words, _ = preprocessing.stem_lem([query], words, [])

    print('Finished Query Preprocessing.')
    return words


if __name__ == '__main__':
    """
    queries = generate_queries(cv_matrix, word2vec, 100, min_length=4, max_length=4)
    print(str(check_valid_queries(queries)))
    utility.save_vector_file("Generated Files/queries.csv", queries)
    """
    queries = utility.load_vector_file("Generated Files/queries.csv")
    res, p_vec = lda_evaluate_query(list(queries.keys())[0], list(queries.values())[0].split(' '), tell=True)
