import itertools
import math
from typing import List
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from tqdm import tqdm
import utility
from models.lda import compute_coherence_values, compute_coherence_values_k_and_priors
from preprocessing import preprocess
import query_handling


def general_grid_search(function, fixed_params, hyper_params, plot=True, y_label="Evaluation Score", save_path=None):
    """
    General function to do a grid-search over another function.
    :param function: function that is searched over. Should return a number if plot is True.
    :param fixed_params: dictionary over the parameters of the function and their values.
    :param hyper_params: dictionary over the hyper-parameters for the function and their list of values to be tested.
    :param plot: boolean indicating whether to plot the results.
    :param y_label: the name of the number returned by the function. Used for plotting.
    :param save_path: the path to save the plot to. Only relevant if plot is True.
    :return: dictionary mapping hyper-parameter combinations to resulting function values.
    """
    # make all combinations of hyper-parameters
    hyper_combs = list(itertools.product(*hyper_params.values()))
    results = []
    for comb in tqdm(hyper_combs):
        # combine names of hyper-parameters with values from the combination
        params = {list(hyper_params.keys())[i]: comb[i] for i in range(len(hyper_params.keys()))}
        # add fixed parameters
        params.update(fixed_params)
        # call function using **kwargs and store result
        res = function(**params)
        with open("generated_files/results.txt", 'a+') as file:
            file.write(str(comb) + ',' + str(res) + '\n')
        results.append(res)
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

    test_combinations = list(itertools.product(Ks, alphas, etas, thresholds))

    if not evaluation:
        # Default sorting is based on K
        test_coherence_combination = list(zip(test_combinations, coherence_values))
        # Sort on alpha values
        # test_combinations = sorted(test_coherence_combination, key = lambda tup: tup[0][1])
        # Sort on eta values
        # test_combinations = sorted(test_coherence_combination, key = lambda tup: tup[0][2])
        # Sort on coherence value
        test_combinations = sorted(test_coherence_combination, key=lambda tup: tup[1], reverse=True)

        combinations_sorted = [x[0] for x in test_combinations]
        coherences_sorted = [x[1] for x in test_combinations]

        plt.plot([str(x) for x in combinations_sorted], coherences_sorted, label="Coherence score")
        plot_settings()
        plt.ylabel("Coherence score")
        save_fig(plot_file_name + ".png")
    else:
        plt.plot([str(x) for x in test_combinations], [x[0][8] for x in tw_eval_results], label="Words without topics")
        plot_settings()
        plt.ylabel("Words without topics")
        save_fig(plot_file_name + "_zero_topic_words" + ".png")

        plt.plot([str(x) for x in test_combinations], [x[1][0][5] for x in tw_eval_results],
                 label="Average words per topic")
        plot_settings()
        plt.ylabel("Average words per topic")
        save_fig(plot_file_name + "_avg_words_per_topic" + ".png")


def plot_settings():
    plt.xticks(rotation=90, fontsize=5)
    plt.xlabel("Combination")
    plt.legend()
    plt.tight_layout()
    plt.grid(1, axis='x')


def save_fig(plot_file_name: str):
    fig = plt.gcf()
    fig.savefig(plot_file_name, dpi=300)
    plt.show()


if __name__ == '__main__':
    model_path = 'LDA/model/document_model'
    words = utility.load_vector_file("generated_files/word2vec.csv")
    documents = list(utility.load_vector_file("generated_files/doc2word.csv").values())
    vocab = Dictionary(documents)
    corpus = [vocab.doc2bow(doc) for doc in documents]

    Ks = [10, 50, 100, 200, 300]
    alphas = [0.5, 0.1, 0.01, 0.0001]
    etas = [0.1, 0.01, 0.001, 0.0001]

    fixed_params = {"model_path": "LDA/model/test_model", "documents": documents, "corpus": corpus, "vocab": vocab}
    hyper_params = {"K": Ks, "alpha": alphas, "eta": etas}
    general_grid_search(query_handling.run_lda_and_compute_perplexity, fixed_params=fixed_params, hyper_params=hyper_params,
                        plot=True, save_path="generated_files/Evaluation/lda_test.png")
