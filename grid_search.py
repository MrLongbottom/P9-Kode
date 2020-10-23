import itertools
import math
from typing import List

import matplotlib.pyplot as plt
from gensim.corpora import Dictionary

from LDA.lda import compute_coherence_values, compute_coherence_values_k_and_priors
from preprocessing import preprocess


def grid_search_coherence():
    start = 2
    step = 3
    cv_matrix, words, texts = preprocess("documents.json")
    limit = math.floor(math.sqrt(cv_matrix.shape[0]))
    dictionary = Dictionary(texts)
    model_list, coherence_values = compute_coherence_values(cv_matrix=cv_matrix,
                                                            dictionary=dictionary,
                                                            texts=texts,
                                                            words=words,
                                                            limit=limit,
                                                            start=start,
                                                            step=step)

    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.show()


def grid_search_coherence_k_and_priors(Ks: List[int], alphas: List[float], etas: List[float], file_name: str = "GridSearch.png"):
    cv_matrix, words, texts = preprocess("documents.json")
    dictionary = Dictionary(texts)
    model_list, coherence_values = compute_coherence_values_k_and_priors(cv_matrix=cv_matrix,
                                                                         dictionary=dictionary,
                                                                         texts=texts,
                                                                         words=words,
                                                                         Ks=Ks,
                                                                         alphas=alphas,
                                                                         etas=etas)

    # Default sorting is based on K
    test_combinations = list(itertools.product(Ks, alphas, etas))
    test_coherence_combination = list(zip(test_combinations, coherence_values))
    # Sort on alpha values
    #test_combinations = sorted(test_coherence_combination, key = lambda tup: tup[0][1])
    # Sort on eta values
    test_combinations = sorted(test_coherence_combination, key = lambda tup: tup[0][2])
    
    combinations_sorted = [x[0] for x in test_combinations]
    coherences_sorted = [x[1] for x in test_combinations]
    
    plt.xticks(rotation=90, fontsize=5)
    plt.plot([str(x) for x in combinations_sorted], coherences_sorted)
    plt.xlabel("Combination")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.tight_layout()
    plt.grid(1, axis='x')
    fig = plt.gcf()
    fig.savefig(file_name, dpi=300)
    plt.show()


if __name__ == '__main__':
    # grid_search_coherence()

    # 4*4*4 = 64 combinations
    #Ks = [10]
    Ks = [10, 40, 80, 160]
    #alphas = [0.01]
    alphas = [0.01, 0.1, 0.3, 0.6]
    #alphas = ['asymmetric']
    #etas = [0.0001]
    etas = [0.0001, 0.001, 0.005, 0.01]
    grid_search_coherence_k_and_priors(Ks, alphas, etas, "GridSearchEta.png")