import itertools
import math
from typing import List

import matplotlib.pyplot as plt
from gensim.corpora import Dictionary

from lda import compute_coherence_values, compute_coherence_values_k_and_priors
from preprocessing import preprocess


def grid_search_coherence():
    """
    Runs a grid search on the coherence value, given a start and step size of K, up to a limit.
    Plots the coherence value of each K evaluated.
    """
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


def grid_search_coherence_k_and_priors(Ks: List[int], alphas: List[float], etas: List[float], thresholds: List[float],
                                       plot_file_name: str = "GridSearch.png", evaluation: bool = False):
    """
    For given lists of Ks, alpha, and etas, calculate the coherence value for each combination of these.
    Plots the coherence values of each combination.
    :param Ks: List of the amount of topics
    :param alphas: List of alpha prior weights
    :param etas: List of eta prior weights
    :param thresholds: List of TW thresholds 
    :param plot_file_name: The file name of the saved figure.
    :param evaluation: Bool for whether the entropy evaluation is to be run
    """
    cv_matrix, words, texts = preprocess("documents.json")
    dictionary = Dictionary(texts)
    _, coherence_values, _, tw_eval_results = compute_coherence_values_k_and_priors(cv_matrix=cv_matrix,
                                                                         dictionary=dictionary,
                                                                         texts=texts,
                                                                         words=words,
                                                                         Ks=Ks,
                                                                         alphas=alphas,
                                                                         etas=etas,
                                                                         thresholds=thresholds,
                                                                         evaluation=evaluation)

    plt.xticks(rotation=90, fontsize=5)
    plt.xlabel("Combination")
    plt.legend()
    plt.tight_layout()
    plt.grid(1, axis='x')
    
    test_combinations = list(itertools.product(Ks, alphas, etas, thresholds))
    
    if not evaluation:
        # Default sorting is based on K
        test_coherence_combination = list(zip(test_combinations, coherence_values))
        # Sort on alpha values
        #test_combinations = sorted(test_coherence_combination, key = lambda tup: tup[0][1])
        # Sort on eta values
        #test_combinations = sorted(test_coherence_combination, key = lambda tup: tup[0][2])
        # Sort on coherence value
        test_combinations = sorted(test_coherence_combination, key = lambda tup: tup[1], reverse = True)

        combinations_sorted = [x[0] for x in test_combinations]
        coherences_sorted = [x[1] for x in test_combinations]

        plt.plot([str(x) for x in combinations_sorted], coherences_sorted, label="Coherence score")
        plt.ylabel("Coherence score")
        save_fig(plot_file_name + ".png")
    else:
        plt.ylabel("Words without topics")
        plt.plot([str(x) for x in test_combinations], [x[0][8] for x in tw_eval_results], label="Words without topics")
        save_fig(plot_file_name + "zero_topic_words" + ".png")
        
        plt.ylabel("Average words per topic")
        plt.plot([str(x) for x in test_combinations], [x[1][0][5] for x in tw_eval_results], label="Average words per topic")
        save_fig(plot_file_name + "avg_words_per_topic" + ".png")

    
def save_fig(plot_file_name: str):
    fig = plt.gcf()
    fig.savefig(plot_file_name, dpi=300)
    plt.show()


if __name__ == '__main__':
    # grid_search_coherence()

    # 4*4*4 = 64 combinations
    Ks = [80]
    #Ks = [10, 40, 80, 160]
    alphas = [0.1]
    #alphas = [0.01, 0.1, 0.3, 0.6]  # 0.1 default from wiki
    #alphas = ['asymmetric']
    #etas = [0.001]
    etas = [0.0001, 0.001, 0.005, 0.01]  # 0.001 default from wiki
    thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.1]  # 0.001 default like the default eta
    grid_search_coherence_k_and_priors(Ks, alphas, etas, thresholds, "GridSearchEval", evaluation=True)
