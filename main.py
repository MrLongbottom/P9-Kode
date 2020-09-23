from LDA.lda import run_lda
from preprocessing import preprocess

if __name__ == '__main__':
    # Setup pipeline for the project
    # Preprocess
    cv_matrix, words, corpus, mini_corpus = preprocess("documents.json")

    # Run LDA
    run_lda(path="./model_test",
            cv_matrix=cv_matrix,
            words=words,
            corpus=corpus)

    # Make Graph
    # graph()

