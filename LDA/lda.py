import json
import math
import pickle

# nltk.download('stopwords')
from sklearn.decomposition import LatentDirichletAllocation as LDA
from tqdm import tqdm

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
    else:
        from multiprocessing import Pool
        with Pool(num_workers) as p:
            p.starmap(train_model, data)


# def count_vectorizer(data):
#     cnt_vectorizer = CountVectorizer(stop_words=stopwords.words('danish'))
#     count_data = cnt_vectorizer.fit_transform(data)
#     return cnt_vectorizer, iter(count_data)
#
#
# def print_topics(model, cnt_vectorizer, n_top_words):
#     words = cnt_vectorizer.get_feature_names()
#     for topic_idx, topic in enumerate(model.components_):
#         print("\nTopic #%d:" % topic_idx)
#         print(" ".join([words[i]
#                         for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == '__main__':
    # Load data
    file_name = "../documents.json"
    length_of_file = get_length_of_file(file_name)
    data = main(file_name)

    num_of_topics = math.floor(math.sqrt(data.shape[1]))
    print(f"number of topics: {num_of_topics}")

    data = iter(data)

    # Model training
    lda = LDA(n_components=num_of_topics, n_jobs=-1)
    itertive_model_train(8, get_length_of_file(file_name), data)
    print("finished")
    with open(f"model", 'wb') as file:
        pickle.dump(lda, file)
