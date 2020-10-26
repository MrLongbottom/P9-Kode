import json
import random
import re
import nltk
import pandas as pd
import scipy.sparse as sparse
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import DanishStemmer
from typing import Dict
from wiktionaryparser import WiktionaryParser


def preprocess(filename_or_docs="documents.json", word_save_filename="Generated Files/word2vec.csv",
               doc_save_filename="Generated Files/doc2vec.csv", doc_word_save_filename="Generated Files/doc2word.csv",
               doc_word_matrix_save_filename="Generated Files/count_vec_matrix.npz", tfidf_matrix_filename = "Generated Files/tfidf_matrix.npz", word_minimum_count=20, word_maximum_doc_percent=0.25,
               doc_minimum_length=20, save=True, word_check=True):
    """
    preprocesses a json file into a docword count vectorization matrix, removing unhelpful words and documents.
    :param filename_or_docs: path of .json file to load (default: "documents.json") or the documents to preprocess
    :param word_save_filename: path of .csv file to save words in vector format. Only relevant if save=True.
    (default: "Generated Files/word2vec.csv")
    :param doc_save_filename: path of .csv file to save documents in vector format. Only relevant if save=True.
    (default: "Generated Files/doc2vec.csv")
    :param doc_word_save_filename: path of .csv file to map documents and contained words using ids.
    (default: "Generated Files/doc2word.csv")
    :param doc_word_matrix_save_filename: path of the .npz file which contains the Count Vectorization matrix.
    (default: "Generated Files/count_vec_matrix.npz")
    :param word_minimum_count: minimum amount a word must be used in the document set to be considered viable
    (default: 20).
    :param word_maximum_doc_percent: maximum percentage of documents that may contain a word for it to be considered
    viable (default: 0.25)
    :param doc_minimum_length: minimum amount of words for a document to be viable (default: 20).
    :param save: boolean indicating whether to save words and document files.
    :param word_check: boolean indicating whether to check words against word databases.
    Can be very slow when using new dataset, but is saved locally afterwards.
    :return: csr-matrix (sparse matrix) containing word frequencies for each document.
    """
    print('Beginning Preprocessing Procedure.')
    step = 1
    # load documents file
    print(f'Step {step}: loading documents.')
    # If filename_or_docs is a string, load documents from path, else continue as if given documents directly
    documents = load_document_file(filename_or_docs) if isinstance(filename_or_docs, str) else filename_or_docs
    # filter documents and create corpus
    documents, corpus = filter_documents(documents, doc_minimum_length)

    # cut off words that are used too often or too little (max/min document frequency) or are stop words
    step += 1
    print(f'Step {step}: stop words and word frequency.')
    words = cut_off_words(corpus, word_maximum_doc_percent, word_minimum_count)

    print(len(words))
    if word_check:
        # cut off words that are not used in danish word databases or are wrong word type
        step += 1
        print(f"Step {step}: word databases and POS-tagging.")
        # TODO possibly replace with real POS tagging, rather than database checks.
        words = word_checker(words)

    # Stemming to combine word declensions
    step += 1
    print(f"Step {step}: Apply Stemming / Lemming")
    corpus, words, documents = stem_lem(corpus, words, documents)

    # filter documents to remove docs that now contain too few words (after all the word filtering)
    step += 1
    print(f"Step {step}: re-filter documents.")
    corpus, documents = refilter_docs(words, corpus, doc_minimum_length, documents)

    # transform documents into a matrix containing counts for each word in each document
    step += 1
    print(f"Step {step}: doc-word matrix construction")
    cv2 = CountVectorizer(vocabulary=words)
    cv_matrix = cv2.fit_transform(corpus)
    print("Matrix is: " + str(cv_matrix.shape))

    # Get new word dict (without the cut words)
    words = value_dictionizer(cv2.get_feature_names())
    # Get new corpus (without the cut words)
    corpus = cv2.inverse_transform(cv_matrix)
    corpus = [list(x) for x in corpus]

    if save:
        step += 1
        print(f'Step {step}: saving files.')
        save_vector_file(word_save_filename, words.values())
        save_vector_file(doc_save_filename, documents.keys())
        save_vector_file(doc_word_save_filename, corpus, seperator='-')
        sparse.save_npz(doc_word_matrix_save_filename, cv_matrix)
        with open("Generated Files/corpus", 'w', encoding='utf8') as json_file:
            json.dump(corpus, json_file, ensure_ascii=False)
    print('Finished Preprocessing Procedure.')
    return cv_matrix, words, corpus


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


def preprocess_query(query: str, word_check=True):
    # cut off words that are used too often or too little (max/min document frequency) or are stop words
    step = 1
    print(f'Step {step}: stop words and word frequency.')
    words = cut_off_words([query], 1.0, 1)

    print(len(words))
    if word_check:
        # cut off words that are not used in danish word databases or are wrong word type
        step += 1
        print(f"Step {step}: word databases and POS-tagging.")
        # TODO possibly replace with real POS tagging, rather than database checks.
        words = word_checker(words)

    # Stemming to combine word declensions
    step += 1
    print(f"Step {step}: Apply Stemming / Lemming")
    corpus, words = stem_lem([query], words)

    print('Finished Query Preprocessing.')
    return words


def cut_off_words(corpus, word_maximum_doc_percent, word_minimum_count):
    nltk.download('stopwords')
    stop_words = stopwords.words('danish')
    cv = CountVectorizer(max_df=word_maximum_doc_percent, min_df=word_minimum_count, stop_words=stop_words)
    cv.fit(corpus)
    words = key_dictionizer(cv.get_feature_names())
    return words


def stem_lem(corpus, words, documents):
    """
    Updates a word list and a corpus to use stemmed words.
    :param corpus: a list of sentences (strings of words separated by spaces)
    :param words: a list of words
    :return: new corpus and words list
    """
    # Stemming
    stemmer = DanishStemmer()
    stop_words = stopwords.words('danish')
    # Update word list to use stemmed words
    translator = {}
    add = []
    remove = []
    for word in tqdm(words):
        stem = stemmer.stem(word)
        if stem != word:
            if word not in remove:
                remove.append(word)
            if stem not in add and stem not in stop_words:
                add.append(stem)
            if word not in translator and stem not in stop_words:
                translator[word] = stem
    words = [x for x in words if x not in remove]
    words.extend([x for x in add if x not in words])

    # update corpus to use stemmed words
    for x in tqdm(range(len(corpus))):
        sen = corpus[x]
        sentence = sen.split(' ')
        for i in range(len(sentence)):
            word = sentence[i]
            if word in translator:
                sentence.append(translator[word])
                sentence.remove(word)
        corpus[x] = ' '.join(sentence)

    return corpus, words, dict(zip(documents.keys(), corpus))


def find_indexes(dict, values):
    for i in tqdm(range(0, len(values))):
        list = []
        for j in values[i]:
            list.append(dict[j])
        values[i] = list
    return values


def key_dictionizer(keys):
    return {y: x for x, y in enumerate(keys)}


def value_dictionizer(values):
    return {x: y for x, y in enumerate(values)}


# TODO make faster? (how fast? sonic fast!)
def cut_corpus(corpus, words):
    cut = []
    words_dict = {x: 0 for x in words}
    for doc in tqdm(corpus):
        sen = []
        for word in doc.split(" "):
            if word in words_dict:
                sen.append(word)
        if len(sen) != 0:
            cut.append(sen)
    return cut


def refilter_docs(words, corpus, doc_minimum_length, documents):
    words_dict = {}
    for word in tqdm(words):
        words_dict[word] = 0
    empty_docs = []
    for doc in tqdm(corpus):
        count = 0
        for word in doc.split(' '):
            if word in words_dict:
                count += 1
                if count >= doc_minimum_length:
                    break
        if count < doc_minimum_length:
            empty_docs.append(doc)
    print("removed " + str(len(empty_docs)) + " docs, " + str(len(corpus) - len(empty_docs)) + " remaining.")
    return [x for x in corpus if x not in empty_docs], {a: b for a,b in documents.items() if b not in empty_docs}


def load_document_file(filename):
    print('Loading documents from "' + filename + '".')
    documents = {}
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            data = json.loads(json_obj)
            documents[data['id']] = data['headline'] + ' ' + data['body']
    print('Loaded ' + str(len(documents)) + ' documents.')
    return documents


def filter_documents(documents, doc_minimum_length):
    print('Filtering documents.')
    corpus = []
    bad_docs = []
    # Reverse dictionary to get unique values only (ie. remove duplicate documents).
    reverse = {v: k for k, v in documents.items()}
    documents = {v: k for k, v in reverse.items()}
    for doc in tqdm(documents.items()):
        text = doc[1]
        text = text.lower()
        text = re.findall(r"[a-zæøå]+", text)
        # Remove documents containing too few words.
        if len(text) < doc_minimum_length:
            bad_docs.append(doc)
            continue
        # reconstruct documents
        text = ' '.join(text)
        documents[doc[0]] = text
        corpus.append(text)
    for doc in bad_docs:
        del documents[doc[0]]
    print('Filtered documents. ' + str(len(documents)) + ' remaining')
    return documents, corpus


def save_vector_file(filename, content, seperator=','):
    """
    Saves content of list as a vector in a file, similar to a Word2Vec document.
    :param filename: path of file to save.
    :param content: list of content to save.
    :return: None
    """
    print('Saving file "' + filename + '".')
    with open(filename, "w") as file:
        for i, c in enumerate(content):
            file.write(str(i) + seperator + str(c) + '\n')
    print('"' + filename + '" has been saved.')


def csv_append(filename, content, index=0):
    """
        Saves content of list as a vector in a file, similar to a Word2Vec document.
        :param filename: path of file to save.
        :param content: list of content to save.
        :return: None
        """
    print('Saving file "' + filename + '".')
    with open(filename, "a") as file:
        id_counter = index
        for c in content:
            file.write(str(id_counter) + "," + str(c) + '\n')
            id_counter += 1
    print('"' + filename + '" has been saved.')


def new_word_db_fetch(words, wik_word_index=0, wik_nonword_index=0):
    # setup Wiktionary Parser

    wik_parser = WiktionaryParser()
    wik_parser.set_default_language('danish')
    wik_parser.RELATIONS = []
    wik_parser.PARTS_OF_SPEECH = ["noun", "verb", "adjective", "adverb", "proper noun"]
    new_words = []
    new_nonwords = []
    for word in tqdm(words):
        try:
            data = wik_parser.fetch(word)
            if len(data) == 0:
                new_nonwords.append(word)
            else:
                new_words.append(word)
        except AttributeError:
            print("something went wrong, with fidning a word on WikWord.")
            continue
    csv_append('NLP/wik_nonwords.csv', new_nonwords, wik_nonword_index)
    csv_append('NLP/wik_words.csv', new_words, wik_word_index)
    return new_words, new_nonwords


def load_word_files(filenames):
    files = []
    for filename in filenames:
        csv_df = pd.read_csv(filename, header=None, encoding='unicode_escape')
        files.append(list(csv_df[1]))
    return files


def word_checker(words):
    files = load_word_files(['NLP/dannet_words.csv', 'NLP/wik_words.csv', 'NLP/wik_nonwords.csv'])
    dannet_words = files[0]
    wik_words = files[1]
    wik_nonwords = files[2]
    wik_remain_words = [v for v in words if v not in wik_words and v not in wik_nonwords]

    # Load new words from fetch databases
    if len(wik_remain_words) != 0:
        print("New words encountered, fetching data.")
        new_words, new_nonwords = new_word_db_fetch(wik_remain_words,
                                                    wik_word_index=len(wik_words), wik_nonword_index=len(wik_nonwords))
        wik_words.extend(new_words)

    # Test how many databases contain the given words
    bad_words = []
    for word in tqdm(words):
        if word not in wik_words and word not in dannet_words:
            bad_words.append(word)
    print(str(len(bad_words)) + " words were not supported by any databases.")
    return [x for x in words if x not in bad_words]


if __name__ == '__main__':
    cv_matrix, words, corpus = preprocess()
    queries = generate_queries(cv_matrix, words, 1000)
