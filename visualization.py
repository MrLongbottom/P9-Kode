from collections import Counter
from os import path

import gensim.corpora as corpora
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from IPython.display import display
from tqdm import tqdm

from grid_search import save_fig
from models.lda import load_lda
from preprocessing import preprocess


def create_doc_main_topic_df(ldamodel, corpus, term_doc_freq):
    """
    Creates a dataframe of each document and its main topic and probability, along with text examples
    :param ldamodel: The LDA model
    :param corpus: The corpus
    :param term_doc_freq: Term document frequency for use with the LDA model
    :return: Dataframe of document main topics, with document text
    """
    doc_topics_df = get_main_topics_df(ldamodel, term_doc_freq)
    doc_topics_df.columns = ['Dominant_Topic', 'Topic_Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(corpus)
    doc_topics_df = pd.concat([doc_topics_df, contents], axis=1)
    return doc_topics_df


def get_main_topics_df(ldamodel, term_doc_freq):
    """
    Creates the document main topic dataframe for each document in the corpus, given by the term_doc_freq
    :param ldamodel: The LDA model
    :param term_doc_freq: Term document frequency for use with the LDA model
    :return: Dataframe of document main topics
    """
    # Init dataframe
    doc_topics_df = pd.DataFrame()
    # Get main topic in each document
    print("Getting main topics...")
    max_ = len(term_doc_freq)  # Amount of documents
    with tqdm(total=max_) as pbar:
        for i, row_list in enumerate(ldamodel[term_doc_freq]):
            # Using the term document frequency on the LDA model gives the topics and probabilities for the documents
            # row_list is a list of the documents topics with probabilities
            row = row_list[0] if ldamodel.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the dominant topic, percentage contribution, and words for each document
            (topic_num, prop_topic) = row[0]
            words_and_probabilities = ldamodel.show_topic(topic_num)  # show_topic default gives 10 words of the topic
            topic_keywords = ", ".join([word for word, prop in words_and_probabilities])
            doc_topics_df = doc_topics_df.append(
                pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            pbar.update()
    print("Finished getting main topics")
    return doc_topics_df


def get_topic_representative_text_dataframe(df_dominant_topics):
    """
    Gets a representative text for each topic from the document with the highest topic contribution.
    :param df_dominant_topics: The dataframe with documents and representative topics
    :return: Dataframe with each topic and representative text from a document
    """
    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100

    df_docs_topics_sorted = pd.DataFrame()
    # Group by topic and sort by topic contribution to documents to get representative document
    df_docs_topics_grouped = df_dominant_topics.groupby('Dominant_Topic')
    for i, grp in df_docs_topics_grouped:
        df_docs_topics_sorted = pd.concat([df_docs_topics_sorted,
                                          grp.sort_values(['Topic_Perc_Contribution'], ascending=False).head(1)],
                                          axis=0)
    # Reset Index
    df_docs_topics_sorted.reset_index(drop=True, inplace=True)

    # Format
    df_docs_topics_sorted.columns = ['Document_No', 'Topic_No', "Topic_Perc_Contribution", "Keywords",
                                     "Representative Text"]
    return df_docs_topics_sorted


def visualization_word_count_for_topic_words(lda_model, corpus, corpus_path, lda_path, topic_start: int, all_topics: bool = True):
    """
    Visualize the word counts and weights for 4 topics from the starting topic number.
    :param lda_model: The LDA model
    :param corpus: The corpus
    :param topic_start: The topic number to start visualizing from
    """
    print("Getting word counts per topic and visualizing")
    topics = lda_model.show_topics(num_topics=0, formatted=False)
    data_flat = [w for w_list in corpus for w in w_list]  # each word in each document
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    print("Word counts calculated")
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    topic_len = 4
    if all_topics:
        topic_len = len(topics)

    # Get max y-axis value for the weight values
    max_y = 0
    for topic in range(topic_start, topic_start + topic_len):
        max_y = max(df.loc[df.topic_id == topic, "importance"].max(), max_y)

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(topic_len, 1, figsize=(8, 5 * topic_len), sharey=True)
    plot_word_counts_and_weights(axes, df, max_y, topic_start)

    fig.tight_layout(w_pad=2)
    fig.subplots_adjust(top=0.95)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22)
    _, model_name, corpus_name = get_save_path(lda_path, corpus_path)
    save_fig("Word count and importance_" + corpus_name + "_" + model_name + "_topic " + str(topic_start) + "-" + str(topic_start + topic_len - 1) + ".pdf")


def plot_word_counts_and_weights(axes, df, max_y, topic_start: int):
    """
    Plots the data from a dataframe intro each topic subplot
    :param axes: The axes/subplots in the plot
    :param df: The dataframe with word count and weight data
    :param max_y: The max y-axis value for the weights
    :param topic_start: The topic number to start visualizing from
    """
    cols = [color for name, color in mcolors.XKCD_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i + topic_start, :], color=cols[i], width=0.5,
               alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i + topic_start, :], color=cols[i],
                    width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, max_y)
        ax.set_title('Topic: ' + str(i + topic_start), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i + topic_start, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')


def visualization_distribution_doc_word_count(lda_model, df_dominant_topics, corpus_path: str = "", lda_path: str = "", topic_start: int = 0):
    """
    Visualizes a distribution of the amount of documents over word counts for the corpus and 4 chosen topics.
    Statistics of this are included in the corpus distribution.
    :param df_dominant_topics: The document dominant topic dataframe
    :param corpus_path: The path to the corpus
    :param topic_start: The topic to start plotting from
    """
    # Get the length of each document
    doc_lens = [len(d) for d in df_dominant_topics.Text]
    max_word_count = max(doc_lens)  # The maximum word count

    # Plots distribution for whole corpus
    plot_word_count_distribution_corpus(corpus_path, doc_lens, max_word_count)
    # Plots distribution for topics
    plot_word_count_distribution_topics(lda_model, df_dominant_topics, max_word_count, topic_start, corpus_path, lda_path)


def plot_word_count_distribution_corpus(corpus_path, doc_lens, max_word_count):
    """
    Plots the distribution of word counts for the corpus.
    :param corpus_path: The path to the corpus
    :param doc_lens: The amount of words in each document
    :param max_word_count: The maximum amount of words in a document
    """
    plt.figure(figsize=(16, 7))
    plt.hist(doc_lens, bins=max_word_count, color='navy')
    # Statistics
    plt.text(max_word_count - 500, 135, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(max_word_count - 500, 120, "Median : " + str(round(np.median(doc_lens))))
    plt.text(max_word_count - 500, 105, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(max_word_count - 500, 90, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(max_word_count - 500, 75, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))
    plt.gca().set(xlim=(0, max_word_count), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, max_word_count, 15).astype(int))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    plt.savefig("Document_word distribution - " + corpus_path.split("/")[-1] + ".pdf")
    plt.show()


def plot_word_count_distribution_topics(lda_model, df_dominant_topics, max_word_count, topic_start, corpus_path, lda_path, all_topics: bool = True):
    """
    Plots the distribution of word counts for 4 topics, starting from the topic_start.
    :param df_dominant_topics: The document dominant topic dataframe
    :param max_word_count: The maximum amount of words in a document
    :param topic_start: The topic to start plotting from
    """
    topics = lda_model.show_topics(num_topics=0, formatted=False)
    topic_len = 4
    if all_topics:
        topic_len = len(topics)

    cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    fig, axes = plt.subplots(topic_len, 1, figsize=(8, 5 * topic_len), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        df_dominant_topic_sub = df_dominant_topics.loc[df_dominant_topics.Dominant_Topic == i + topic_start, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins=max_word_count, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sb.kdeplot(doc_lens, color="black", fill=False, ax=ax.twinx())  # makes a smooth distribution curve
        ax.set(xlim=(0, max_word_count), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: ' + str(i + topic_start), fontdict=dict(size=16, color=cols[i]))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.xticks(np.linspace(0, max_word_count, 15).astype(int))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    _, model_name, corpus_name = get_save_path(lda_path, corpus_path)
    save_fig("Word count distribution per topic_" + corpus_name + "_" + model_name + "_topic " + str(topic_start) + "-" + str(topic_start + topic_len - 1) + ".pdf")


def get_save_path(model_path: str, corpus_path: str):
    lda_model_name = model_path.split("/")[-1]
    corpus_name = corpus_path.split("/")[-1]
    return "generated_files/df_" + lda_model_name + "_" + corpus_name + ".pkl", lda_model_name, corpus_name


def create_or_load_doc_topic_dataframe(lda_model, corpus, tdf, lda_path, corpus_path):
    df_save_path, _, _ = get_save_path(lda_path, corpus_path)

    if path.exists(df_save_path):
        print("Dataframe file exists")
        df_topic_docs_keywords = pd.read_pickle(df_save_path)
    else:
        print("Creating dataframe file...")
        df_topic_docs_keywords = create_doc_main_topic_df(ldamodel=lda_model, corpus=corpus, term_doc_freq=tdf)
        df_topic_docs_keywords.to_pickle(df_save_path)
        print("Dataframe file created")

    # Format
    df_dominant_topics = df_topic_docs_keywords.reset_index()
    df_dominant_topics.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contribution', 'Keywords', 'Text']
    return df_dominant_topics


if __name__ == '__main__':
    lda_path = "LDA/model/final_model(30, 0.1, 0.1)(30, 0.1, 0.1)"
    corpus_name = "corpus2017"
    lda_model = load_lda(lda_path)

    cv_matrix, vocab, documents = preprocess("2017_data.json")

    # Create Dictionary
    id2word = corpora.Dictionary(documents)
    # Create Corpus: Term Document Frequency
    tdf = [vocab.doc2bow(text) for text in documents]

    # Create or load main dataframe file
    df_dominant_topics = create_or_load_doc_topic_dataframe(lda_model, documents, tdf, lda_path, corpus_name)
    display(df_dominant_topics.head(10))

    # Get a representative text for each topic from the document with the highest topic contribution
    df_topic_representative_text = get_topic_representative_text_dataframe(df_dominant_topics)
    display(df_topic_representative_text.head(10))

    # Visualize word count and word weights per topic
    visualization_word_count_for_topic_words(lda_model, documents, corpus_name, lda_path, topic_start=0)

    # Visualize the distribution of amount of words over documents
    visualization_distribution_doc_word_count(lda_model, df_dominant_topics, corpus_name, lda_path, topic_start=0)
