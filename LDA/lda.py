import math
from typing import Dict

from gensim import matutils
from gensim.models import LdaModel, LdaMulticore
from gensim.test.utils import datapath
from scipy.sparse import csr_matrix

from preprocessing import preprocess


def fit_lda(data: csr_matrix, vocab: Dict):
    """
    Fit LDA from a scipy CSR matrix (data).
    :param data: a csr_matrix representing the vectorized words
    :param vocab: a dictionary over the words
    :return: a lda model trained on the data and vocab
    """
    print('fitting lda...')
    return LdaMulticore(matutils.Sparse2Corpus(data, documents_columns=False),
                        id2word=vocab,
                        num_topics=math.floor(math.sqrt(data.shape[1])))


def save_lda(lda: LdaModel, path: str):
    # Save model to disk.
    temp_file = datapath(path)
    lda.save(temp_file)


def load_lda(path: str):
    return LdaModel.load(path)


def word_cloud(corpus):
    # Import the wordcloud library
    from wordcloud import WordCloud
    # Join the different processed titles together.
    long_string = ','.join(corpus)
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image().show()


if __name__ == '__main__':
    # Loading data and preprocessing
    model_path = '/home/simba/Documents/P9/P9-Kode/LDA/model/model_18'
    data, cv = preprocess('2018_data.json')
    vocab = dict([(i, s) for i, s in enumerate(cv.get_feature_names())])

    # Fitting the model and saving it
    lda_model = fit_lda(data, vocab)
    save_lda(lda_model, model_path)

    # Loading the model
    lda = load_lda(model_path)
