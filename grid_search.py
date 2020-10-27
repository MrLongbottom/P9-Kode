import itertools
import math
from typing import List
from test import test
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from tqdm import tqdm

from LDA.lda import compute_coherence_values, compute_coherence_values_k_and_priors, run_lda
from preprocessing import preprocess


def general_grid_search(function, fixed_params, hyper_params, plot=True, y_label="Evaluation Score", save_path=None):
    # make all combinations of hyper-parameters
    hyper_combs = list(itertools.product(*hyper_params.values()))
    results = []
    for comb in tqdm(hyper_combs):
        # combine names of hyper-parameters with values from the combination
        params = {list(hyper_params.keys())[i]: comb[i] for i in range(len(hyper_params.keys()))}
        # add fixed parameters
        params.update(fixed_params)
        # call function using **kwargs and store result
        results.append(function(**params))
    # plot results
    if plot:
        plt.plot([str(x) for x in hyper_combs], results)
        plt.xticks(rotation=90, fontsize=6)
        plt.tight_layout()
        plt.grid(1, axis='x')
        plt.ylabel(y_label)
        plt.xlabel(f"({str(list(hyper_params.keys()))[1:-1]})")
        if save_path is not None:
            fig = plt.gcf()
            fig.savefig(save_path)
        plt.show()

    return {hyper_combs[i]: results[i] for i in range(len(results))}


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


def grid_search_coherence_k_and_priors(Ks: List[int], alphas: List[float], etas: List[float], plot_file_name: str = "GridSearch.png"):
    """
    For given lists of Ks, alpha, and etas, calculate the coherence value for each combination of these.
    Plots the coherence values of each combination.
    :param Ks: List of the amount of topics
    :param alphas: List of alpha prior weights
    :param etas: List of eta prior weights
    :param plot_file_name: The file name of the saved figure. Should include a filetype (e.g. '.png')
    """
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
    fig.savefig(plot_file_name)
    plt.show()


if __name__ == '__main__':
    test = general_grid_search(test, {"param1": 234, "param3": 156}, {"param2": [1, 2, 3], "param4": ["absdasd", "safasfagf", 123135]}, save_path="Generated Files/test.png")
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