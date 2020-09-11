import re
import json

from multiprocessing import Pool
from tqdm import tqdm

# ID, word, count
word2vec = [[], [], []]
id_counter = 0
documents = []
# name of file to load words from
load_filename = "documents.json"
# name of file to save down
save_filename = "word2vec.txt"
pool_size = 4


def main():
    print('Beginning Word2Vec Procedure.')
    print('Beginning to load documents from "' + load_filename + '".')
    load_file(load_filename)
    print('Finished Loaded Word2Vec from "' + load_filename + '".')

    wordcounts = [[], []]
    for doc in tqdm(documents):
        test2, test3 = word2vec_doc_load(doc)
        wordcounts[0] += (test2)
        wordcounts[1] += (test3)

    id_counter = 0
    for i in tqdm(range(0, len(wordcounts[0]))):
        if wordcounts[0][i] not in word2vec[1]:
            word2vec[0].append(id_counter)
            word2vec[1].append(wordcounts[0][i])
            word2vec[2].append(wordcounts[1][i])
        else:
            word2vec[2][word2vec[1].index(wordcounts[0][i])] += wordcounts[1][i]

    print('Found ' + str(len(word2vec[0])) + " unique words.")

    print('Beginning to save Word2Vec in "' + save_filename + '".')
    save_file(save_filename)
    print('Finished Saved Word2Vec in "' + save_filename + '".')

    print('Finished Word2Vec Procedure.')


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            data = json.loads(json_obj)
            documents.append(data)


def save_file(filename):
    with open(filename, "w") as file:
        for i in range(0, len(word2vec[0])):
            file.write(str(word2vec[0][i]) + ", " + word2vec[1][i] + '\n')


def word2vec_doc_load(doc):
    string = doc['headline'] + " " + doc['body']
    string.lower()
    # basic word regex filter
    words = re.findall(r"[a-zæøå]+", string)
    unique_words = []
    count = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
            count.append(1)
        else:
            count[unique_words.index(word)] += 1
    return unique_words, count


if __name__ == "__main__":
    main()
