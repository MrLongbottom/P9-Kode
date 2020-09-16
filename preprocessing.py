import json
import re
import nltk
import csv

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm
from nltk.corpus import stopwords
from wiktionaryparser import WiktionaryParser

nltk.download('stopwords')


def preprocess(load_filename="documents.json", word_save_filename="Generated Files/word2vec.csv",
               doc_save_filename="Generated Files/doc2vec.csv", word_minimum_count=20, word_maximum_doc_percent=0.25,
               doc_minimum_length=20, save=True):
    """
    preprocesses a json file into a doc_word count matrix, removing unhelpful words and documents
    :param load_filename: path of .json file to load (default: "documents.json")
    :param word_save_filename: path of .txt file to save words in vector format. Only relevant if save=True
    (default: "Generated Files/word2vec.csv")
    :param doc_save_filename: path of .txt file to save documents in vector format. Only relevant if save=True
    (default: "Generated Files/doc2vec.csv")
    :param word_minimum_count: minimum amount of words for a document to be viable (default: 20).
    :param word_maximum_doc_percent: maximum percentage of documents that may contain a word for it to be considered
    viable (default: 0.25)
    :param doc_minimum_length: minimum amount a word must be used in the documents to be considered viable.
    :param save: boolean indicating whether to save words and document files.
    :return: csr-matrix (sparse matrix) containing word frequencies for each document.
    """
    print('Beginning Word2Vec Procedure.')

    # load documents file
    print('Step 1: loading documents.')
    documents = load_document_file(load_filename)
    # filter documents and create corpus
    documents, corpus = filter_documents(documents, doc_minimum_length)

    # cut off words that are used too often or too little (max/min document frequency) or are stop words
    print('Step 2: stop words and word frequency')
    stop_words = stopwords.words('danish')
    cv = CountVectorizer(max_df=word_maximum_doc_percent, min_df=word_minimum_count, stop_words=stop_words)
    cv.fit(corpus)

    # cut off words that are not used in danish word databases or are wrong word type
    print("Step 3: word databases and POS-tagging")
    words = cv.get_feature_names()
    words = word_checker(words)

    # transform documents into a matrix containing counts for each word in each document
    print("Step 4: doc-word matrix construction")
    cv2 = CountVectorizer(vocabulary=words)
    X = cv2.fit_transform(corpus)
    print("Matrix is: " + str(X.shape))

    # calculate term frequency - inverse document frequency
    # (might not be needed)
    tf = TfidfTransformer()
    X2 = tf.fit_transform(X)

    if save:
        print('Saving word and document lookup files.')
        save_vector_file(word_save_filename, words)
        save_vector_file(doc_save_filename, documents.keys())
    return X, words


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
    for doc in tqdm(documents.items()):
        text = doc[1]
        text = text.lower()
        text = re.findall(r"[a-zæøå]+", text)
        # Remove documents containing too few words.
        if len(text) < doc_minimum_length:
            bad_docs.append(doc)
            continue
        text = ' '.join(text)
        documents[doc[0]] = text
        corpus.append(text)
    for doc in bad_docs:
        del documents[doc[0]]
    print('Filtered documents. ' + str(len(documents)) + ' remaining')
    return documents, corpus


def save_vector_file(filename, content):
    """
    Saves content of list as a vector in a file, similar to a Word2Vec document.
    :param filename: path of file to save.
    :param content: list of content to save.
    :return: None
    """
    print('Saving file "' + filename + '".')
    with open(filename, "w") as file:
        id_counter = 0
        for c in content:
            file.write(str(id_counter) + ", " + str(c) + '\n')
            id_counter += 1
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
            file.write(str(id_counter) + ", " + str(c) + '\n')
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
        data = wik_parser.fetch(word)
        if len(data) == 0:
            new_nonwords.append(word)
        else:
            new_words.append(word)
    csv_append('NLP/wik_nonwords.csv', new_nonwords, wik_nonword_index)
    csv_append('NLP/wik_words.csv', new_words, wik_word_index)
    return new_words, new_nonwords


def load_word_files(filenames):
    files = []
    for filename in filenames:
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            content = []
            for row in csv_reader:
                content.append(row[1][1:])
            files.append(content)
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
                                                    wik_word_index=len(wik_words), wik_nonword_index= len(wik_nonwords))
        wik_words.extend(new_words)

    # Test how many databases contain the given words
    bad_words = []
    for word in tqdm(words):
        if word not in wik_words and word not in dannet_words:
            bad_words.append(word)
    print(str(len(bad_words)) + " words were not supported by any databases.")
    return [x for x in words if x not in bad_words]


if __name__ == '__main__':
    preprocess()
