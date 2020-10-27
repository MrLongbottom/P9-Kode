import math
from gensim.corpora import Dictionary

from lda import run_lda
from preprocessing import preprocess

if __name__ == '__main__':
    # Setup pipeline for the project
    # Preprocess
    cv_matrix, words, corpus = preprocess("data.json")

    # Run LDA
    K = math.floor(math.sqrt(cv_matrix.shape[0]) / 2)
    params = (K, None, 0.001)

    run_lda("LDA/model/full_model" + str(params),
            cv_matrix=cv_matrix,
            words=words,
            corpus=corpus,
            dictionary=Dictionary(corpus),
            save_path="Generated Files/",
            param_combination=params,
            tw_threshold=0.001,
            dt_threshold=0.025)
