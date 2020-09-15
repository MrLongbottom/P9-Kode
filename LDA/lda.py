import json
import math
import numpy as np

from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from preprocessing import main


def get_file_generator(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        for json_obj in json_file:
            yield json.loads(json_obj)['body']


def get_length_of_file(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        return sum(1 for line in json_file)


def train_model(slice):
    lda.partial_fit(slice.toarray())


def itertive_model_train(num_workers: int, num_samples: int, data):
    if num_workers == 1:
        for sample in tqdm(range(num_samples)):
            slice = next(data).toarray()
            lda.partial_fit(slice)
        return lda
    else:
        from multiprocessing import Pool
        with Pool(num_workers) as p:
            p.starmap(train_model, data)


def count_vectorizer(data):
    cnt_vectorizer = CountVectorizer(stop_words=stopwords.words('danish'))
    count_data = cnt_vectorizer.fit_transform(data)
    return cnt_vectorizer, iter(count_data)


def print_topics(model, cnt_vectorizer, n_top_words):
    words = cnt_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == '__main__':
    # Load data
    file_name = "../documents.json"
    length_of_file = get_length_of_file(file_name)
    data = iter(main(file_name))
    # num of topics
    num_of_topics = math.floor(math.sqrt(get_length_of_file(file_name)))
    print(f"number of topics: {num_of_topics}")

    # Model training
    lda = LDA(n_components=num_of_topics, n_jobs=-1)
    model = itertive_model_train(4, get_length_of_file(file_name), data)
    print("finished")
