import itertools
import json
import math
import pickle

# nltk.download('stopwords')
from concurrent.futures.process import ProcessPoolExecutor

from sklearn.decomposition import LatentDirichletAllocation as LDA
from tqdm import tqdm

from preprocessing import preprocess


def get_file_generator(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            yield json.loads(json_obj)['body']


def get_length_of_file(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        return sum(1 for line in json_file)


def train_model(slice):
    print("started training model")
    lda.partial_fit(slice.toarray())


def itertive_model_train(num_workers: int, num_samples: int, data):
    if num_workers == 1:
        for sample in tqdm(range(num_samples)):
            slice = next(itertools.islice(data, 10)).toarray()
            lda.partial_fit(slice)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executer:
            executer.map(train_model, itertools.islice(data, 2))


if __name__ == '__main__':
    # Load data
    file_name = "../documents.json"
    length_of_file = get_length_of_file(file_name)
    data = preprocess(file_name)

    num_of_topics = math.floor(math.sqrt(data.shape[1]))
    print(f"number of topics: {num_of_topics}")

    data = iter(data)

    # Model training
    lda = LDA(n_components=num_of_topics, n_jobs=-1)
    itertive_model_train(1, get_length_of_file(file_name), data)
    print("finished")
    with open(f"model", 'wb') as file:
        pickle.dump(lda, file)
