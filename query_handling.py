import itertools
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

import lda
import preprocessing
import utility

doc2word = utility.load_vector_file("Generated Files/doc2word.csv")
word2vec = utility.load_vector_file("Generated Files/word2vec.csv")
inverse_w2v = {v: k for k, v in word2vec.items()}


def evaluate_query_doc(function, query: List[str],
                       document_index: int, matrices: Tuple[sp.csc_matrix, sp.csc_matrix]):
    """
    Evaluate a query based on a function and document index
    :param matrices: document topic and topic word matrices
    :param document_index: the document index
    :param function: the evaluation function
    :param query: the list of query words
    :return: the product of the evaluate function
    """
    p_wd = []
    for word in query:
        word_index = inverse_w2v[word]
        p_wd.append(function(document_index, word_index, matrices))
    return np.prod(p_wd)


def evaluate_query(function, query_index, query_words, matrices, tell=False):
    """
    Evaluating a query based on a function given and the query
    which consists of query index and words
    :param matrices: docment-topic and topic word matrices
    :param function: the evaluation function you want to use
    :param query_index: the query's document index
    :param query_words: the words in the query
    :param tell: do you want to print top 3 and the words after it has finished
    :return: the index of the query in the ranked list and the list it self.
    """
    lst = {}
    with Pool(processes=8) as p:
        max_ = len(doc2word)
        with tqdm(total=max_) as pbar:
            for i, score in enumerate(
                    p.starmap(partial(evaluate_query_doc, function, query_words),
                              list(enumerate(itertools.repeat(matrices, times=max_))))):
                lst[i] = score
                pbar.update()

    sorted_list = utility.rankify(lst)
    if tell:
        print(f"query: {query_words}")
        print(f"index of document: {sorted_list.index(query_index)}")
        print(f"number 1 index: {sorted_list[0]} number 1: words {doc2word[sorted_list[0]]}\n")
        print(f"number 2 index: {sorted_list[1]} number 2: words {doc2word[sorted_list[1]]}\n")
        print(f"number 3 index: {sorted_list[2]} number 3: words {doc2word[sorted_list[2]]}\n")
        print(f"real document: {doc2word[query_index]}")
    return sorted_list.index(query_index), lst


def lda_runthrough_query(model_path, cv, words, mini_corpus, K, alpha, eta):
    lda_model, dt_matrix, tw_matrix = lda.run_lda(
        model_path,
        cv,
        words,
        mini_corpus,
        Dictionary(mini_corpus),
        "Generated Files/",
        (K, alpha, eta))
    return lda_model.log_perplexity(dt_matrix.shape[0])
# results = []
# result_matrix = np.matmul(dt_matrix.A, tw_matrix.A)

# for query in tqdm(queries):
#     res, p_vec = grid_lda_evaluate(query, result_matrix)
#     results.append(res)
# return np.mean(results)


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
        query_length = random.randrange(min_length, max_length + 1)
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
    queries = generate_queries(cv_matrix, word2vec, 10, min_length=1, max_length=4)
    print(str(check_valid_queries(queries)))
    utility.save_vector_file("Generated Files/queries.csv", queries)

    # queries = utility.load_vector_file("Generated Files/queries.csv")
    # res, p_vec = lda_evaluate_query(list(queries.keys())[0], list(queries.values())[0].split(' '), tell=True)
