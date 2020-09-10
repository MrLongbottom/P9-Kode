import re
import json

from tqdm import tqdm

print('Beginning Word2Vec Procedure.')

# name of file to load words from
documents = []

filename = "documents.json"
with open(filename, "r", encoding="utf-8") as json_file:
    for json_obj in json_file:
        data = json.loads(json_obj)
        documents.append(data)

# ID, word, count
word2vec = [[], [], []]
id_counter = 0
for doc in tqdm(documents):
    string = doc['headline'] + " " + doc['body']
    string.lower()
    # basic word regex filter
    words = re.findall(r"[A-ZÆØÅa-zæøå]+", string)
    # collect id's, unique words, and word counts
    for word in words:
        if word not in word2vec[1]:
            word2vec[0].append(id_counter)
            word2vec[1].append(word)
            word2vec[2].append(1)
            id_counter += 1
        else:
            word2vec[2][word2vec[1].index(word)] += 1

print('Found ' + str(len(word2vec[0])) + " unique words.")

# name of file to save down
filename = "word2vec.txt"
with open(filename, "w") as file:
    for i in range(0, len(word2vec[0])):
        file.write(str(word2vec[0][i]) + ", " + word2vec[1][i] + '\n')
print('Saved Word2Vec in "' + filename + '".')

print('Finished Word2Vec Procedure.')
