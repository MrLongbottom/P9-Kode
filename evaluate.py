import seaborn as sb
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import scipy.sparse as sp


def evaluate_distribution_matrix(dis_matrix: sp.spmatrix, show: bool = True, tell: bool = True, save_path: str = None,
                                 column_name: str = "A", row_name: str = "B"):
    """
    Evaluate document-topic distribution matrix, involving a combination of:
    * printing statistics
    * showing boxplots
    * pruning empty docs and topics, and pruning topics that are too common
    :param dis_matrix: distribution matrix to be evaluated.
    :param column_name: name of columns for printing
    :param row_name: name of rows for printing
    :param show: whether to show boxplots
    :param tell: whether to print statistics
    :param save_path: path of file to save, default is None, meaning no saving
    :return: potentially pruned matrix
    """
    sb.set_theme(style="whitegrid")
    return_stats = []
    stat_names = ["Zeros", "Minimums", "Maximums", "Averages", "Medians", "Entropies"]
    # loop over A-B distribution, then B-A distribution
    for ab in range(2):
        zeros, empties, avgs, maxs, mins, medians, entropies = [], [], [], [], [], [], []
        # Fill out statistics for each row/column
        max_loop = 1 if ab == 0 else 0
        for i in tqdm(range(0, dis_matrix.shape[max_loop])):
            vec = dis_matrix.getcol(i) if ab == 0 else dis_matrix.getrow(i)
            non_vec = vec.nonzero()[ab]
            zeros.append((vec.shape[ab] - len(non_vec))/vec.shape[ab])
            avgs.append(vec.mean())
            maxs.append(vec.max())
            mins.append(vec.min())
            medians.append(np.median(vec.toarray()))
            if len(non_vec) == 0:
                empties.append(i)
            else:
                vec_array = vec.toarray().T[0] if ab == 0 else vec.toarray()[0]
                entropies.append(entropy(vec_array, base=vec.shape[ab]))
        # Print statistics
        if tell:
            if ab == 0:
                print(f"{column_name}-{row_name} Distributions.")
            else:
                print(f"{row_name}-{column_name} distributions.")
            print(f"{len(empties)} empty vectors")
        stats = {stat_names[0]: zeros, stat_names[1]: mins, stat_names[2]: maxs, stat_names[3]: avgs,
                 stat_names[4]: medians, stat_names[5]: entropies}
        # Make stats ready for return
        for name, stat in stats.items():
            return_stats.append(stats_of_list(stat, name=name, tell=tell))
        # Save stats
        if save_path is not None:
            with open(save_path, "w+") as f:
                for name, stat in zip(stats.keys(), return_stats):
                    f.write(f"{name}, "+", ".join(str(x) for x in stat)+"\n")

        # TODO reimplement boxplots
        #if show:
        #    ax = sb.boxplot(x=zeros)
        #    plt.show()
    return return_stats


def stats_of_list(list, name: str = "List", tell: bool = True):
    zeros = (len(list) - np.count_nonzero(list))/len(list)
    mini = min(list)
    maxi = max(list)
    avg = np.mean(list)
    medi = np.median(list)
    entro = 1 if np.isnan(entropy(list, base=len(list))) else entropy(list, base=len(list))
    if tell:
        print(f"{name} Statistics (length: {len(list)}).")
        print(f"Zeros percentage in {name}: {zeros}")
        print(f"Minimum {name}: {mini}")
        print(f"Maximum {name}: {maxi}")
        print(f"Average {name}: {avg}")
        print(f"Median {name}: {medi}")
        print(f"Entropy {name}: {entro}\n")
    return [zeros, mini, maxi, avg, medi, entro]

if __name__ == '__main__':
    print()