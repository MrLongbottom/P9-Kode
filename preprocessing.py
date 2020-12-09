import json
import random
import re

import gensim
import nltk
import pandas as pd
import scipy.sparse as sparse
import numpy as np
import utility

import lemmy
from matplotlib import pyplot
import seaborn as sb
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import DanishStemmer
from typing import Dict
from wiktionaryparser import WiktionaryParser


def preprocess(filename_or_docs="documents.json", word_save_filename="Generated Files/word2vec.csv",
               doc_save_filename="Generated Files/doc2vec.csv", doc_word_save_filename="Generated Files/doc2word.csv",
               doc_word_matrix_save_filename="Generated Files/count_vec_matrix.npz", word_minimum_count=20,
               word_maximum_doc_percent=0.25,
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
    document_ids = load_document_file(filename_or_docs) if isinstance(filename_or_docs, str) else filename_or_docs
    # filter documents and create document set
    document_ids, documents = filter_documents(document_ids, doc_minimum_length)

    # cut off words that are used too often or too little (max/min document frequency) or are stop words
    step += 1
    print(f'Step {step}: stop words and word frequency.')
    # Filter words based on rarity
    vocab = gensim.corpora.Dictionary(documents)
    vocab.filter_extremes(word_minimum_count, word_maximum_doc_percent)
    # Get stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('danish')
    stop_words.extend(list(utility.load_vector_file("NLP/stopwords.csv").values()))
    # Remove stopwords
    words = vocab.token2id
    bad_ids = []
    for sw in stop_words:
        if sw in words:
            bad_ids.append(words[sw])
    vocab.filter_tokens(bad_ids=bad_ids)
    vocab.compactify()

    if word_check:
        # cut off words that are not used in danish word databases or are wrong word type
        step += 1
        print(f"Step {step}: word databases and POS-tagging.")
        words = vocab.token2id
        bad_ids = word_checker(words)
        for sw in stop_words:
            if sw in words:
                bad_ids.append(words[sw])
        vocab.filter_tokens(bad_ids=bad_ids)
        vocab.compactify()

    # Stemming to combine word declensions
    step += 1
    print(f"Step {step}: Apply Stemming / Lemming")
    words = list(vocab.token2id.keys())
    vocab, documents = stem_lem(words, documents, stem_or_lem=False)

    for id, x in enumerate(documents):
        test = vocab.doc2idx(x)
        documents[id] = [x[i] for i in range(len(x)) if test[i] != -1]

    # transform documents into a matrix containing counts for each word in each document
    step += 1
    print(f"Step {step}: doc-word matrix construction")
    words = list(vocab.token2id.keys())
    cv = CountVectorizer(vocabulary=words)
    cv_matrix = cv.fit_transform([' '.join(x) for x in documents])
    print("Matrix is: " + str(cv_matrix.shape))

    if save:
        step += 1
        print(f'Step {step}: saving files.')
        utility.save_vector_file(word_save_filename, words)
        utility.save_vector_file(doc_save_filename, document_ids.keys())
        utility.save_vector_file(doc_word_save_filename, [' '.join(x) for x in documents])
        vocab.save("Generated Files/vocab")
        sparse.save_npz(doc_word_matrix_save_filename, cv_matrix)
    print('Finished Preprocessing Procedure.')
    return cv_matrix, vocab, documents


def cut_off_words(corpus, word_maximum_doc_percent, word_minimum_count, use_tfidf: bool = False):
    nltk.download('stopwords')
    stop_words = stopwords.words('danish')
    stop_words.extend(list(utility.load_vector_file("NLP/stopwords.csv").values()))
    if not use_tfidf:
        cv = CountVectorizer(max_df=word_maximum_doc_percent, min_df=word_minimum_count, stop_words=stop_words)
        cv_matrix = cv.fit_transform(corpus)
        words = key_dictionizer(cv.get_feature_names())
        return words
    else:
        cv = CountVectorizer(stop_words=stop_words)
        cv_matrix = cv.fit_transform(corpus)
        words = key_dictionizer(cv.get_feature_names())
        words = {v: k for k,v in words.items()}
        words = filter_tfidf(cv_matrix, words)

        return words


def stem_lem(words, documents, stem_or_lem: bool = False):
    """
    Updates a word list and a corpus to use stemmed words.
    :param stem_or_lem: bool indicating whether to apply stemming or lemmatizer. True is stem, False is lem.
    :param corpus: a list of sentences (strings of words separated by spaces)
    :param words: a list of words
    :return: new corpus and words list, were all words have been replaced by stemmed/lemmetized versions.
    """
    stop_words = stopwords.words('danish')
    stop_words.extend(list(utility.load_vector_file("NLP/stopwords.csv").values()))
    if stem_or_lem:
        # Stemming
        stemmer = DanishStemmer()
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
    else:
        lemmer = lemmy.load("da")
        # build up dictionary that translates old words into their new versions
        translator = {}
        add = []
        remove = []
        for word in tqdm(words):
            lem = lemmer.lemmatize("", word)
            other = [x for x in lem if x != word]
            if len(other) > 0:
                if word not in lem and word not in remove:
                    remove.append(word)
                # add all lem options if they are not stopwords
                add.extend([x for x in lem if x not in stop_words and x not in add])
                if word not in translator and lem not in stop_words:
                    lem = " ".join(lem)
                    translator[word] = lem
        words = [x for x in words if x not in remove]
        words.extend([x for x in add if x not in words])

    # update corpus to use stemmed words
    for x in tqdm(range(len(documents))):
        sentence = documents[x]
        for i in range(len(sentence)):
            word = sentence[i]
            if word in translator:
                sentence[i] = translator[word]
        sentence = ' '.join(sentence)
        sentence = sentence.split(' ')
        documents[x] = sentence

    diction = gensim.corpora.Dictionary(documents)
    d_words = diction.token2id
    good_ids = [d_words[x] for x in words]
    diction.filter_tokens(good_ids=good_ids)
    diction.compactify()

    return diction, documents


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
    return [x for x in corpus if x not in empty_docs], {a: b for a, b in documents.items() if b not in empty_docs}


def load_document_file(filename):
    print('Loading documents from "' + filename + '".')
    documents = {}
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            data = json.loads(json_obj)
            documents[data['id']] = data['headline'] + ' ' + data['body']
    print('Loaded ' + str(len(documents)) + ' documents.')
    return documents


def cal_tf_idf(cv):
    n_vec = np.bincount(cv.indices, minlength=cv.shape[1])
    idf_vec = np.log10(cv.shape[0] / n_vec)
    #cv.data = np.log10(1+cv.data)

    tf_idf_matrix = cv.multiply(idf_vec)
    return sparse.csc_matrix(tf_idf_matrix)


def filter_tfidf(cv, words):
    tf_idf = cal_tf_idf(cv)
    rare_thresh = 0.0015
    common_thresh = 1.5
    total_mean = tf_idf.mean(axis=0)
    total_mean = np.where(total_mean < rare_thresh, 0, total_mean)
    indices1 = total_mean.nonzero()[1]
    data_mean = np.array([tf_idf.getcol(i).data.mean() for i in range(tf_idf.shape[1])])
    data_mean = np.where(data_mean < common_thresh, 0, data_mean)
    indices2 = np.array(np.nonzero(data_mean)[0])
    words = [words[i] for i in np.intersect1d(indices1, indices2)]
    words = {words[i]: i for i in range(len(words))}
    return words


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
        corpus.append(text)
        text = ' '.join(text)
        documents[doc[0]] = text
    for doc in bad_docs:
        del documents[doc[0]]
    print('Filtered documents. ' + str(len(documents)) + ' remaining')
    return documents, corpus


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
    bad_ids = []
    for word in tqdm(words):
        if word not in wik_words and word not in dannet_words:
            bad_ids.append(words.get(word))
    print(str(len(bad_ids)) + " words were not supported by any databases.")
    return bad_ids


if __name__ == '__main__':
    """
    words = utility.load_vector_file("Generated Files/word2vec.csv")
    docs = list(utility.load_vector_file("Generated Files/doc2word.csv").values())
    words2 = {}
    for doc in docs:
        for word in doc:
            words2[word] = 0
    words3 = gensim.corpora.Dictionary(docs)
    """

    print("Hello World!")
    cv_matrix, vocab, documents = preprocess()
