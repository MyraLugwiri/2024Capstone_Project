# importing all the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import google.generativeai as genai
import nltk
import streamlit as st
import plotly.express as px

nltk.download('stopwords')
nltk.download('punkt')

# libraries for loading environment variables
from dotenv import load_dotenv
import os


##******* Connection to bigQuery **********
# libraries to access data from storage (bigquery)
# from google.cloud import bigquery

# project_id = 'upbeat-cargo-413117'
# dataset_id = 'labelled_covid_tweets'
# table_id = 'labelled-tweets'
#
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "data/upbeat-cargo-413117-67fd6c3cf2ec.json"
# # bigquery client
# client = bigquery.Client()
#
# # reference to the table
# table_ref = client.dataset(dataset_id).table(table_id)
#
# # getting the table data
# table = client.get_table(table_ref)
#
# for field in table.schema:
#     print("field name:", field.name)
#     print("field type:", field.field_type)
##******* End of Connection to bigQuery **********

def gemini_config():
    """Configuring the gemini API"""
    load_dotenv()
    api_key = os.getenv('API_KEY')
    genai.configure(api_key=api_key)
    return genai


# function to read the data
@streamlit.cache_data
def reading_data():
    """Read in data from storage"""
    labelled_tweets = pd.read_csv('data/COVID-19 labeled tweets.csv')
    unlabelled_tweets = pd.read_csv("data/tweets_dataset_diseases.csv")
    youtube_data = pd.read_csv('data/comments_collected.csv')

    sample_test = pd.read_csv('data/test_data.csv')
    sample_test['Year'] = sample_test['Year'].astype(int)
    sample_test['Year'] = sample_test['Year'].map(lambda x: '{:04d}'.format(x))
    topics_data = [sentence for sentence in labelled_tweets['text']]
    topics_data = [sentence for sentence in unlabelled_tweets['text']] + topics_data
    topics_data = [sentence for sentence in youtube_data['text']] + topics_data
    return topics_data, sample_test


@streamlit.cache_data
def topic_modelling(documents):
    """Generate a list of topics present in a document"""
    # Tokenization and stopword removal
    stop_words = set(stopwords.words('english'))
    tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
    tokenized_documents = [[word for word in doc if word.isalnum() and word not in stop_words] for doc in
                           tokenized_documents]

    dictionary = corpora.Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=5)
    return topics


@streamlit.cache_resource
def generate_topic_labels(topic):
    """Generate topic labels using Google's Gemini AI"""
    global result, recommendations
    while True:
        try:
            load_dotenv(dotenv_path='.env')
            api_key = os.getenv('API_KEY')
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('models/gemini-pro')
            for word in topic:
                prompt = (
                             'can you provide labels for the for the topics identified in topic modelling in the list '
                             'attached,'
                             'the list contains topics 0 through 4 and the keywords associated with the said topics ') + str(
                    word)
                result = model.generate_content(prompt)

                prompt1 = (
                              'can you generate a list of recommendations to help deal with the key issues arising '
                              'from the topics provided'
                              'the topics are gathered from users social media live comments on health') + str(result)
                recommendations = model.generate_content(prompt1)
            return result, recommendations
        except Exception as error:
            print('The error is:', error)
            continue


def generate_recommendations():
    try:
        model = gemini_config()
    except Exception as error:
        print('The error is:', error)
    pass


def sentiment_scores_graph(data):
    """Plots the sentiment scores' Zscores"""
    mean_sentiment_scores = data.groupby(['date'])['Sentiment_Score_Zscore'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    # plt.plot(data['date'], data['Sentiment_Score_Zscore'], marker='o', linestyle='-')
    plt.plot(mean_sentiment_scores['date'], mean_sentiment_scores['Sentiment_Score_Zscore'], marker='o', linestyle='-')
    plt.title('Sentiment Scores Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score (Z-score)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    return plt


def graphs(df_selection):
    """Plots line graph, boxplot and histogram using the data containing sentiments"""
    mean_sentiment_scores = df_selection.groupby(['date', 'label'])['score'].mean().reset_index()

    # Creating the line graph, boxplot and a histogram
    fig_time_series = px.line(mean_sentiment_scores, x='date', y='score', color='label',
                              title='Sentiment Score Over Time')
    fig_distribution = px.box(df_selection, x='label', y='score', title='Sentiment Score Distribution by Label')
    fig_frequency = px.histogram(df_selection, x='score', title='Sentiment Score Frequency Distribution')

    left, right, center = st.columns(3)
    left.plotly_chart(fig_time_series, use_container_width=True)
    right.plotly_chart(fig_distribution, use_container_width=True)
    center.plotly_chart(fig_frequency, use_container_width=True)


def word_cloud_formation(topic_output):
    word_frequencies = {}
    for topic_id, topic_text in topic_output:
        # Extract the frequency values and word strings
        terms = topic_text.split(' + ')
        for term in terms:
            freq, word = term.split('*')
            word = word.strip('"')  # Remove quotes from word string
            freq = float(freq)  # Convert frequency value to float
            # Add frequency to word_frequencies dictionary
            if word in word_frequencies:
                word_frequencies[word] += freq
            else:
                word_frequencies[word] = freq

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)

    # Plot word cloud
    # plt.figure(figsize=(15, 6))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.title('Word Cloud of Topic Modelling Output')
    # plt.tight_layout()
    # plt.show()
    #
    # sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
    # words, frequencies = zip(*sorted_word_frequencies)
    #
    # # Plot the bar graph
    # plt.subplot(1, 2, 2)
    # plt.bar(words, frequencies)
    # plt.xlabel('Words')
    # plt.ylabel('Frequency')
    # plt.title('Word Frequency Graph')
    # plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    # plt.tight_layout()
    # plt.show()
    df_bar = pd.DataFrame(word_frequencies.items(), columns=['Words', 'Frequency'])

    # Create the word cloud figure
    fig_wordcloud = px.imshow(wordcloud)
    fig_wordcloud.update_layout(title='Word Cloud of Topic Modelling Keywords')

    # Create the bar graph figure
    fig_bar = px.bar(df_bar, x='Words', y='Frequency', title='Word Frequency Distribution',
                     labels={'Words': 'Words', 'Frequency': 'Frequency'})
    fig_bar.update_xaxes(tickangle=45)

    return fig_wordcloud, fig_bar