import itertools
import random
from functools import partial
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
import lda
from typing import Dict, List
import pandas
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from scipy.spatial import distance
import preprocessing
import utility
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
from lda import get_document_topics_from_model, load_lda

doc2word = utility.load_vector_file("Generated Files/doc2word.csv")
word2vec = utility.load_vector_file("Generated Files/word2vec.csv")
inverse_w2v = {v: k for k, v in word2vec.items()}
cv_matrix = sp.load_npz("Generated Files/count_vec_matrix.npz")


def evaluate_query_doc(function, query: List[str], document_index: int):
    """
    Evaluate a query based on a function and document index
    :param function: the evaluation function
    :param query: the list of query words
    :param document_index: the index of the document
    :return: the product of the evaluate function
    """
    p_wd = []
    for word in query:
        word_index = inverse_w2v[word]
        p_wd.append(function(document_index, word_index))
    return np.prod(p_wd)


def evaluate_query(function, query_index, query_words, tell=False):
    """
    Evaluating a query based on a function given and the query
    which consists of query index and words
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
                    p.imap(partial(evaluate_query_doc, function, query_words), range(0, max_))):
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


def lda_runthrough_query(queries, model_path, cv, words, mini_corpus, K, alpha, eta):
    _, dt_matrix, tw_matrix = lda.run_lda(
        model_path,
        cv,
        words,
        mini_corpus,
        Dictionary(mini_corpus),
        "Generated Files/",
        (K, alpha, eta))
    results = []
    for query in queries.items():
        res, p_vec = evaluate_query(query[0], query[1].split(' '), tell=False)
        results.append(res)
        print("query done")
    return np.mean(results)


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


def make_personalization_vector(word: str, topic_doc_matrix, corpus: Dictionary, lda: LdaModel):
    """
    Making a personalization vector based on the topics.
    Gives each documents in the document a score based on the distribution value
    :param lda: the lda model
    :param corpus: a dictionary over the whole document set
    :param word: a word
    :param topic_doc_matrix: topic document matrix
    :return: np.ndarray
    """
    topics = get_document_topics_from_model([word], lda, corpus)
    vector = np.zeros(topic_doc_matrix.shape[0])
    for key, value in topics.items():
        vector[topic_doc_matrix.getrow(key).nonzero()[1]] = value
    return vector


def make_personalization_vector_word_based(word: str, topic_doc_matrix, lda):
    """
    Takes a query and transforms it into a vector based on each words topic distribution
    For each word we find its topic distribution and compare it against every other document
    using jensen shannon distance.
    :param word: a word
    :param topic_doc_matrix: topic document matrix
    :param lda: the lda model
    :return: a personalization vector (np.ndarray)
    """
    # Initialization
    p_vector = np.zeros(topic_doc_matrix.shape[0])
    vector = np.zeros(topic_doc_matrix.shape[1])
    df = pandas.read_csv("Generated Files/word2vec.csv", header=None)
    words = dict(zip(df[0], df[1]))
    topic_word_matrix = lda.get_topics()

    # Getting the word index and then getting the topic distribution for that given word
    word_index = [key for key, value in words.items() if value == word][0]
    vector += topic_word_matrix.T[word_index]

    for index, doc in enumerate(topic_doc_matrix):
        p_vector[index] = 1 - distance.jensenshannon(vector, doc.toarray()[0])
    return p_vector


def query_topics(query: List[str], model_path: str, topic_doc_path, corpus) -> np.ndarray:
    """
    Takes a list of words and makes a personalization vector based on these
    :param corpus: the corpus of the whole document set
    :param query: list of words
    :param model_path: lda model path
    :param topic_doc_path: topic document matrix path
    :return: a personalization vector
    """

    lda = load_lda(model_path)
    topic_doc_matrix = sp.load_npz(topic_doc_path)[:2000]
    p_vector = np.zeros(2000)
    for word in query:
        if word in corpus.values():
            p_vector += make_personalization_vector_word_based(word, topic_doc_matrix, lda)
        else:
            # Todo needs to be cut
            p_vector += make_personalization_vector(word, topic_doc_matrix, corpus, lda)
    return p_vector / np.linalg.norm(p_vector)


def query_expansion(query: List[str], n_top_word: int = 10) -> List[str]:
    """
    Expands a given query based on the word in the query
    Specifically for each word we find the frequency of word before and after
    and add n_top_word most frequent word to the query.
    :param query: List of words
    :param n_top_word: number of frequent word you want to add.
    :return: expanded query
    """
    documents = utility.load_vector_file("Generated Files/doc2word.csv")
    doc_id = query[0]
    result = []
    words = query[1]
    for word in words.split(' '):
        expanded_query = {}
        # append original word to query
        document_ids = [ids for ids, values in documents.items() if word in values]
        for new_id_doc in document_ids:
            # add window size neighboring words
            document = documents[new_id_doc]
            word_index = document.index(word)
            if word_index == 0:
                before_word = document[word_index]
            elif word_index == len(document) - 1:
                after_word = document[word_index]
            else:
                before_word = document[word_index - 1]
                after_word = document[word_index + 1]
        expanded_query[before_word] = expanded_query.get(before_word, 0) + 1
        expanded_query[after_word] = expanded_query.get(after_word, 0) + 1
        sorted_query_words = list(dict(sorted(expanded_query.items(), key=lambda x: x[1], reverse=True)).keys())
        result.append(sorted_query_words[:n_top_word])
    result.append(words.split(' '))
    return list(set(itertools.chain.from_iterable(result)))


def query_run_with_expansion():
    """
    A running example of the expanded query and yield the topic distribution
    for each query generated.
    :return: 
    """
    vectorizer = sp.load_npz("Generated Files/tfidf_matrix.npz")
    words = utility.load_vector_file("Generated Files/word2vec.csv")
    dictionary = Dictionary([words.values()])
    queries = generate_queries(vectorizer, words, 10, 4)
    expanded_queries = []

    for query in tqdm(list(queries.items())):
        expanded_queries.append(query_expansion(query, 5))

    lda = load_lda("LDA/model/document_model(83, None, 0.001)")
    topic_distributions = []
    for exp_query in expanded_queries:
        topic_distributions.append(
            get_document_topics_from_model(exp_query, lda, dictionary, lda.num_topics))

    for query, topic_dis in zip(expanded_queries, topic_distributions):
        print(f"Topic distribution: {topic_dis}")


if __name__ == '__main__':
    queries = generate_queries(cv_matrix, word2vec, 10, min_length=1, max_length=4)
    print(str(check_valid_queries(queries)))
    utility.save_vector_file("Generated Files/queries.csv", queries)

    # queries = utility.load_vector_file("Generated Files/queries.csv")
    # res, p_vec = lda_evaluate_query(list(queries.keys())[0], list(queries.values())[0].split(' '), tell=True)
