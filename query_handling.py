import itertools
import random
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm

import preprocessing
import utility
from models import lda
from models.lda import get_document_topics_from_model, load_lda

doc2word = utility.load_vector_file("generated_files/doc2word.csv")
word2vec = utility.load_vector_file("generated_files/word2vec.csv")
inverse_w2v = {v: k for k, v in word2vec.items()}


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


def lda_runthrough_query(model_path, documents, corpus, vocab, K, alpha, eta):
    lda_model = lda.run_lda(model_path,
                            documents,
                            corpus,
                            vocab,
                            "Generated Files/",
                            (K, alpha, eta))
    return lda_model.log_perplexity(corpus)


def evaluate_queries(queries, evaluation_function):
    results = []
    for query in tqdm(queries):
        if type(query[1]) == str:
            p_vec = evaluation_function([query[1]])
            results.append(p_vec)
        else:
            p_vec = evaluation_function(query[1])
            results.append(p_vec)
    return results

  
def generate_document_queries(count_matrix, words: Dict[int, str], count: int, min_length: int = 1,
                              max_length: int = 4):
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


def generate_topic_queries(count_matrix,
                           document_topic_matrix,
                           words: Dict[int, str],
                           count: int,
                           min_length: int = 1,
                           max_length: int = 4):
    """
    Generates queries for random topics based and samples from 1-4 documents from that topic
    based on the topic distribution values.
    It then adds the highest tfidf word to the query.
    If that word is already in the query it takes the next one in the tf-idf list.
    :param count_matrix: CountVectorization matrix
    :param document_topic_matrix: dt matrix
    :param words: words dictionary
    :param count: number of queries wanted
    :param min_length: min words per query (exact length is random)
    :param max_length: max words per query (exact length is random)
    :return: dictionary mapping document ids to queries
    """
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(count_matrix)
    queries = []
    topic_count = document_topic_matrix.shape[1]
    for i in tqdm(range(count)):
        topic_id = random.randrange(0, topic_count)
        query_length = random.randrange(min_length, max_length + 1)
        query = []
        topic_vec = tfidf_matrix.getcol(topic_id)
        document_ids_in_topic = topic_vec.nonzero()[0]
        normalized_topic_vector = normalize(topic_vec[document_ids_in_topic].toarray(), norm="l1", axis=0)
        sampled_doc_ids = [np.random.choice(document_ids_in_topic, p=normalized_topic_vector[:, 0]) for x in
                           range(0, query_length)]
        for document in sampled_doc_ids:
            index = 0
            tf_idf_list = tfidf_matrix.getrow(document).toarray()[0].argsort()[-max_length:][::-1]
            tf_idf_word = words[tf_idf_list[index]]
            while tf_idf_word in query:
                index += 1
                tf_idf_word = words[tf_idf_list[index]]
            query.append(tf_idf_word)
        query = ' '.join(query)
        queries.append((topic_id, query))
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
    result = []
    for word in query:
        expanded_query = {}
        # append original word to query
        document_ids = [ids for ids, values in documents.items() if word in values]
        for new_id_doc in document_ids:
            # add window size neighboring words
            document = documents[new_id_doc]
            word_index = document.index(word)
            if word_index == 0:
                before_word = document[word_index]
                expanded_query[before_word] = expanded_query.get(before_word, 0) + 1
            elif word_index == len(document) - 1:
                after_word = document[word_index]
                expanded_query[after_word] = expanded_query.get(after_word, 0) + 1
            else:
                before_word = document[word_index - 1]
                after_word = document[word_index + 1]
                expanded_query[before_word] = expanded_query.get(before_word, 0) + 1
                expanded_query[after_word] = expanded_query.get(after_word, 0) + 1
        sorted_query_words = list(dict(sorted(expanded_query.items(), key=lambda x: x[1], reverse=True)).keys())
        result.append(sorted_query_words[:n_top_word])
    result.append(query)
    return list(set(itertools.chain.from_iterable(result)))


def query_run_with_expansion():
    """
    A running example of the expanded query and yield the topic distribution
    for each query generated.
    :return: 
    """
    vectorizer = sp.load_npz("generated_files/tfidf_matrix.npz")
    words = utility.load_vector_file("generated_files/word2vec.csv")
    dictionary = Dictionary([words.values()])
    queries = generate_document_queries(vectorizer, words, 10, 4)
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
    # Loading matrices
    cv_matrix = sp.load_npz("Generated Files/count_vec_matrix.npz")
    dt_matrix = sp.load_npz("Generated Files/(30, 0.1, 0.1)topic_doc_matrix.npz")
    word2vec = utility.load_vector_file("Generated Files/word2vec.csv")

    for i in range(1, 5):
        # Doc query generation
        queries = generate_document_queries(cv_matrix, word2vec, 80, min_length=i, max_length=i)
        # Doc query expansion
        queries = {a: ' '.join(query_expansion(b.split(' '))) for a, b in queries.items()}
        utility.save_vector_file(f"queries/exp_doc_queries{i}.csv", queries)
        print(queries)

        # Topic query generation
        queries = generate_topic_queries(cv_matrix, dt_matrix, word2vec, 80, min_length=i, max_length=i)
        # Topic query expansion
        queries = [(a, ' '.join(query_expansion(b.split(' ')))) for a, b in queries]
        utility.save_vector_file_nonunique(f"queries/exp_top_queries{i}.csv", queries)
        print(queries)
