import re, numpy as np, pandas as pd
import scipy.sparse as sp
import gensim.corpora as corpora
from tqdm import tqdm
from IPython.display import display
import os.path
from os import path
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter

from lda import load_lda, load_corpus
from grid_search import save_fig


def create_doc_main_topic_df(ldamodel, corpus, term_doc_freq):
    """
    Creates a dataframe of each document and its main topic and probability, along with text examples
    :param ldamodel: The LDA model
    :param corpus: The corpus
    :param term_doc_freq: Term document frequency for use with the LDA model
    :return: Dataframe of document main topics, with document text
    """
    doc_topics_df = get_main_topics_df(ldamodel, term_doc_freq)
    doc_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

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


def visualization_word_count_for_topic_words(lda_model, corpus, topic_start):
    print("Getting word counts per topic and visualizing")
    topics = lda_model.show_topics(num_topics=0, formatted=False)
    data_flat = [w for w_list in corpus for w in w_list]  # each word in each document
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])
            
    print("Word counts calculated")
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
    
    max_y = 0
    for topic in range(topic_start, topic_start+4):
        max_y = max(df.loc[df.topic_id==topic, "importance"].max(), max_y)

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i + topic_start, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i + topic_start, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, max_y)
        ax.set_title('Topic: ' + str(i + topic_start), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i + topic_start, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.subplots_adjust(top=0.9)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22)    
    save_fig("Word count and importance_topic " + str(topic_start) + "-" + str(topic_start+3) + ".png")

    
def get_save_path_df_as_pickle(model_path: str, corpus_path: str):
    lda_model_name = model_path.split("/")[-1]
    corpus_name = corpus_path.split("/")[-1]
    return "Generated Files/df_" + lda_model_name + "_" + corpus_name + ".pkl"


def create_or_load_dataframe(df_save_path, lda_model, corpus, tdf):
    if path.exists(df_save_path):
        print("Dataframe file exists")
        df_topic_sents_keywords = pd.read_pickle(df_save_path)
    else:
        print("Creating dataframe file...")
        df_topic_sents_keywords = create_doc_main_topic_df(ldamodel=lda_model, corpus=corpus, term_doc_freq=tdf)
        df_topic_sents_keywords.to_pickle(df_save_path)
        print("Dataframe file created")
    return df_topic_sents_keywords


if __name__ == '__main__':
    lda_path = "LDA/model/document_model"
    corpus_path = "Generated Files/corpus2017"
    lda_model = load_lda(lda_path)
    corpus = load_corpus(corpus_path)

    # Create Dictionary
    id2word = corpora.Dictionary(corpus)
    # Create Corpus: Term Document Frequency
    tdf = [id2word.doc2bow(text) for text in corpus]

    # Create of get main dataframe file
    df_save_path = get_save_path_df_as_pickle(lda_path, corpus_path)
    df_topic_docs_keywords = create_or_load_dataframe(df_save_path, lda_model, corpus, tdf)
    
    # Format
    df_dominant_doc_topic = df_topic_docs_keywords.reset_index()
    df_dominant_doc_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    display(df_dominant_doc_topic.head(10))

    # Visualize word count and word weights per topic
    visualization_word_count_for_topic_words(lda_model, corpus, topic_start=8)
