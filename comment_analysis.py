import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
# detecting anomalies using GAMLSS model
from scipy.stats import gamma
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# data aggregation libraries
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

# from haystack import document_stores
# import csv
# from haystack import DocumentStore
# from haystack.document_stores import InMemoryDocumentStore
import topic_modelling


def loading_model(text):
    output = []
    classifier = pipeline("sentiment-analysis", model="checkpoint-63-20240207T122927Z-001/checkpoint-63")
    # classifier = pipeline("sentiment-analysis", model="/content/drive/MyDrive/result/checkpoint-63")
    output = classifier(text)
    return output


def gemini_model_chat(prompt):
    """"Will provide chat functionalities connected to the GEMINI API and archive data"""
    genai = topic_modelling.gemini_config()
    model = genai.GenerativeModel('models/gemini-pro')

    convo = model.start_chat(history=[
        {
            "role": "user",
            "parts": [
                "you are a health chat bot that will assess social media data for people's perception, feelings, "
                "opinions and identify what are the infectious disease symptoms people are talking about in their "
                "social media posts alot. you will also provide summaries of the said chats"]
        },
        {
            "role": "model",
            "parts": [
                "**User**: Hey health bot, I'm feeling really sick. Can you help me figure out what's "
                "wrong?\n\n**Health Bot**: Hi there! I can try to help you assess your symptoms. First, "
                "tell me a little bit about how you're feeling.\n\n**User**: I have a sore throat, headache, "
                "and cough. I'm also feeling really tired and achy.\n\n**Health Bot**: Okay, those are all common "
                "symptoms of a cold or flu. Have you had any other symptoms, like fever, chills, or shortness of "
                "breath?\n\n**User**: No, I haven't had any of those.\n\n**Health Bot**: That's good. It sounds like "
                "you may have a mild cold. I recommend getting plenty of rest, drinking lots of fluids, and taking "
                "over-the-counter medications to help relieve your symptoms. If your symptoms get worse or you start "
                "to develop new ones, you should see a doctor.\n\n**User**: Thanks for the advice. I'll definitely do "
                "that.\n\n**Summary:** The user is experiencing symptoms of a cold, such as a sore throat, headache, "
                "cough, and fatigue. The bot advised the user to get plenty of rest, drink lots of fluids, "
                "and take over-the-counter medications to help relieve their symptoms. The bot also recommended that "
                "the user see a doctor if their symptoms get worse or they start to develop new ones.\n\n**User**: "
                "I've been seeing a lot of people on social media talking about the flu. Is it really that bad this "
                "year?\n\n**Health Bot**: Yes, the flu is particularly bad this year. According to the Centers for "
                "Disease Control and Prevention (CDC), the flu has already caused an estimated 8.7 million illnesses, "
                "85,000 hospitalizations, and 5,900 deaths in the United States.\n\n**User**: Wow, that's a lot! What "
                "are the symptoms of the flu?\n\n**Health Bot**: The symptoms of the flu can vary, but they typically "
                "include fever, cough, sore throat, runny nose, body aches, headache, and fatigue. Some people may "
                "also experience vomiting and diarrhea.\n\n**User**: How can I protect myself from getting the "
                "flu?\n\n**Health Bot**: The best way to protect yourself from getting the flu is to get a flu shot. "
                "The flu shot is a vaccine that helps your body develop immunity to the flu virus. Other ways to "
                "protect yourself include washing your hands frequently, avoiding contact with people who are sick, "
                "and staying home from work or school if you are sick.\n\n**Summary:** The user is concerned about "
                "the flu and wants to know how to protect themselves. The bot explained that the flu is particularly "
                "bad this year and provided information about the symptoms of the flu and how to prevent it. The bot "
                "also recommended that the user get a flu shot.\n\n**User**: I've been feeling really tired and achy "
                "lately. I'm also having trouble sleeping. Could it be a sign of an infectious disease?\n\n**Health "
                "Bot**: It's possible. Fatigue and aches can be symptoms of a number of infectious diseases, "
                "including the flu, strep throat, and mononucleosis. Trouble sleeping can also be a sign of an "
                "infection.\n\n**User**: What other symptoms should I look out for?\n\n**Health Bot**: Other symptoms "
                "of infectious diseases can include fever, cough, sore throat, runny nose, headache, and nausea. If "
                "you're experiencing any of these symptoms, it's important to see a doctor to get a diagnosis and "
                "start treatment.\n\n**User**: I'll definitely do that. Thanks for the advice.\n\n**Summary:** The "
                "user is experiencing symptoms of an infectious disease, such as fatigue, aches, and trouble "
                "sleeping. The bot provided information about other symptoms to look out for and recommended that the "
                "user see a doctor to get a diagnosis and start treatment."]
        },
    ])
    convo.send_message(prompt)
    return convo


def data_aggregation(df):
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    # Fit ARIMA model
    model = ARIMA(train['score'], order=(5, 1, 0))
    model_fit = model.fit()
    model_summary = model_fit.summary()

    forecast_values = model_fit.forecast(steps=len(test))
    # Forecast
    # forecast, stderr, conf_int = model_fit.forecast(steps=len(test))
    forecast = forecast_values.iloc[0]
    stderr = forecast_values.iloc[1]
    conf_int = forecast_values.iloc[2]
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train['date'], train['score'], label='Training data')
    plt.plot(test['date'], test['score'], label='Test data')
    # plt.plot(test['date'], forecast, label='Forecast', color='red')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.title('ARIMA Model Forecast')
    plt.legend()
    plt.show()

    # Fit Isolation Forest model
    model = IsolationForest(contamination=0.05)  # Adjust contamination parameter as needed
    model.fit(df[['score']])

    # Predict outliers (anomalies)
    outliers = model.predict(df[['score']])

    # Identify outlier indices
    outlier_indices = np.where(outliers == -1)[0]

    # Print outlier indices and corresponding sentiment scores
    print("Outlier Indices:", outlier_indices)
    print("Outlier Sentiment Scores:", df.iloc[outlier_indices]['score'])

    return forecast_values, outliers


def anomaly_detection(social_media_data):
    social_media_data['date'] = pd.to_datetime(social_media_data['date'])
    social_media_data['Sentiment_Score_Zscore'] = zscore(social_media_data['score'])

    # Set anomaly threshold (e.g., Z-score threshold of 2)
    anomaly_threshold = 1.5

    # Identify anomalies
    anomalies = social_media_data[social_media_data['Sentiment_Score_Zscore'].abs() > anomaly_threshold]
    print('anomalies', anomalies)
    ### trial code
    mean_sentiment_scores = social_media_data.groupby(['date'])['score'].mean().reset_index()
    mean_anomalies_scores = anomalies.groupby(['date'])['score'].mean().reset_index()
    ### trial code
    # Plot sentiment scores over time with anomalies
    plt.figure(figsize=(10, 6))
    # plt.plot(social_media_data.index, social_media_data['score'], label='score')
    # plt.plot(anomalies.index, anomalies['score'], color='red', label='Anomaly', linestyle='None', marker='o')
    plt.plot(mean_sentiment_scores['date'], mean_sentiment_scores['score'], label='score')
    plt.plot(mean_anomalies_scores['date'], mean_anomalies_scores['score'], color='red', label='Anomaly', linestyle='None', marker='o')
    plt.title('Sentiment Scores Over Time with Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.grid(True)

    # # Display detected anomalies
    return plt, anomalies


def outbreak_prediction(social_media_data):
    # Prepare data for training
    X = social_media_data[['score']]
    # We'll create a placeholder target variable for demonstration purposes
    y = [0] * len(social_media_data)  # Assuming no disease outbreaks are indicated in the absence of labeled data

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier (or any other suitable model)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance (accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    return accuracy


def fit_gamlss(data):
    def weibull_neg_log_likelihood(params):
        mu = params[0]
        sigma = params[1]
        shape = params[2]
        return -np.sum(gamma.logpdf(data, a=shape, loc=0, scale=mu / shape))

    #Initial guess for parameters
    initial_guess = [data.mean(), data.std(), 1]

    #Minimizing negative log-likelihood
    result = minimize(weibull_neg_log_likelihood, initial_guess, method='L-BFGS-B')
    #Extracting parameters
    mu = result.x[0]
    sigma = result.x[1]
    shape = result.x[2]

    return mu, sigma, shape


def detect_anomalies(data, sigma):
    """Detecting anomalies from the score using Weighted moving averages """
    threshold = 1.5
    # computing exponentially weighted moving averages
    ewma = pd.Series(data['score']).ewm(span=30).mean()
    # computing residuals
    residuals = np.abs(data['score'] - ewma)

    # Identifying anomalies
    anomalies = data['score'][residuals > threshold * sigma]

    # Identifying anomalies indices
    anomaly_indices = residuals[residuals > threshold * sigma].index

    # Getting records of anomalies
    anomaly_records = data.loc[anomaly_indices]

    return anomalies, anomaly_records


def plot_data_with_anomalies(data, anomalies):
    """plotting the anomalies detected"""
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['score'], label='Accuracy')
    # plt.plot(anomalies['date'], anomalies['score'], color='red', label='Anomalies')
    plt.scatter(anomalies.index, anomalies['score'], color='red', label='Anomalies', zorder=5)
    # plt.plot(anomalies['date'], anomalies, color='red', label='Anomalies')
    plt.title('Accuracy Over Time with Anomalies')
    plt.xlabel('Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


def merge_ground_truth(social_media_data, ground_truth_data):
    """Merging ground truth values with the anomalies detected"""
    ground_truth_data['date'] = pd.to_datetime(ground_truth_data['date'])
    # merging ground truth data with sentiment score data
    combined_data = pd.merge(social_media_data, ground_truth_data, on='date', how='left')
    return combined_data


def calculate_metrics(ground_truth, detected_anomalies, tolerance=1e-6):
    """Calculating the precision, recall and f1_score of the anomalies detected"""
    intersection = ground_truth['score'].isin(detected_anomalies)
    intersecting_rows = ground_truth[intersection]

    true_positives = sum(
        1 for anomaly in intersecting_rows['score'] if abs(float(anomaly) - float(anomaly)) < tolerance)
    false_positives = len(detected_anomalies) - true_positives
    false_negatives = len(ground_truth) - true_positives

    if true_positives + false_positives == 0:
        precision = 0  # caters for division by zero
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
