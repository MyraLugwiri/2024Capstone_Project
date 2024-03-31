# Necessary libraries for Topic Modelling
import nltk
import pandas as pd
import streamlit as st
from st_pages import Page, show_pages
from streamlit_extras.metric_cards import style_metric_cards

import comment_analysis
from page_design import home_page

# library for data visualisation

nltk.download('stopwords')
nltk.download('punkt')
import topic_modelling


def selected_date_data(data, dates):
    filtered_data = data[data['Year'].isin(dates)]
    return filtered_data


def main():
    st.set_page_config(page_title="Home", page_icon="🏥", layout="wide")
    st.markdown(home_page, unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            .image-container {
                box-shadow: 2px 2px 6px 0px rgba(0, 0, 0, 0.3);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # app pages
    show_pages(
        [
            Page('frontend.py', "Home", "🏠"),
            Page('pages/1_Query_Data.py', 'Query', '❓')
        ]
    )
    st.title('Infectious Disease Monitor')
    # st.image('images/fusion-medical-animation-rnr8D3FNUNY-unsplash.jpg')
    # sidebar
    st.sidebar.header('Health Monitor')
    st.sidebar.image('images/Blue_Minimalist_Medical_Logo-removebg-preview.png')
    st.markdown("""
    #### :bell: Trends in infectious disease
    Provide the social media data collected in order to identify current trends in infectious diseases 
    using user's sentiments. The process will involve predicting the sentiment present in each individual statement 
    and then we will have recommendations generated by GEMINI on what to do based on the data provided.
    """)

    # Reading sample data
    # topics_data, sample_test_data = topic_modelling.reading_data()
    #
    # # date selector
    # date_selector = st.multiselect('Date Filter', options=sample_test_data['Year'].unique())
    # selected_dates = [str(date) for date in date_selector]
    # filtered_sample_data = selected_date_data(sample_test_data, selected_dates)
    #
    # st.write(filtered_sample_data)

    # plt, new_anomalies = comment_analysis.anomaly_detection(filtered_sample_data)
    # st.pyplot(plt)

    # Main comment analysis to predict infectious disease outbreak
    comment_analysed = st.text_area(label='Enter the data here')
    if st.button(label='Submit Data'):
        comments = comment_analysed.split('\n')
        comment_analysed_list = []
        comment_analysed_list.append(comments)

        for comment in comment_analysed_list:
            dates = [sublist[:10] for sublist in comment_analysed_list[0]]
            dates_df = pd.DataFrame({'date': dates})
            comment_predicts = comment_analysis.loading_model(comment)
            # creating a dataframe from containing the identified sentiments and the score
            comment_predicts_df = pd.DataFrame(comment_predicts)

            # predicting what the future sentiments will look like
            result_df = pd.concat([dates_df, comment_predicts_df], axis=1)
            forecast_prediction, model_summary = comment_analysis.data_aggregation(result_df)

            # st.write('forecast prediction', forecast_prediction)

            # identifying anomalies present in the data
            plt, new_anomalies = comment_analysis.anomaly_detection(result_df)

            # **** Start of Computed Analytics ****

            #  Displaying the data in the Dataframe **
            with st.expander("VIEW DATASET"):
                showData = st.multiselect('Filter: ', result_df.columns, default=['date', 'label', 'score', 'Sentiment_Score_Zscore'])
                st.dataframe(result_df[showData], use_container_width=True)
            new_anomalies['score'] = pd.to_numeric(new_anomalies['score'], errors='coerce')
            new_anomalies['Sentiment_Score_Zscore'] = pd.to_numeric(new_anomalies['Sentiment_Score_Zscore'], errors='coerce')
            # compute top analytics
            score_total = (new_anomalies['Sentiment_Score_Zscore']).sum()
            score_mode = (new_anomalies['score']).mode().iloc[0]
            score_mean = (new_anomalies['score']).mean()
            score_median = (new_anomalies['score']).median()
            # rating = float(pd.Series(df_selection['Rating']).sum())

            total1, total2, total3, total4 = st.columns(4, gap='small')
            with total1:
                st.info('Sum Sentiment Score', icon="📈")
                st.metric(label="Sum Score", value=f"{score_total:.02f}")

            with total2:
                st.info('Mode Sentiment Score', icon="🎯")
                st.metric(label="Mode Score", value=f"{score_mode*100:.02f}%")

            with total3:
                st.info('Average Sentiment Score', icon="🔍")
                st.metric(label="Average Score", value=f"{score_mean*100:.02f}%")

            with total4:
                st.info('Central Sentiment Score', icon="📊")
                st.metric(label="Median Score", value=f"{score_median*100:.02f}%")
            style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",box_shadow=True)

            # **** End of Computed Analytics ****

            # **** Start Plotting the statistics for Anomalies identified ****
            topic_modelling.graphs(new_anomalies)
            # **** Start Plotting the statistics for Anomalies identified ****
            st.pyplot(plt)
            # predicting an infectious disease outbreak using the labelled  data
            predicted_outbreak = comment_analysis.outbreak_prediction(result_df)

            # reading the data
            topic_data = topic_modelling.reading_data()

            modelled_data = topic_modelling.topic_modelling(comment)
            wordcloud, wordbar = topic_modelling.word_cloud_formation(modelled_data)
            left, right = st.columns(2)
            left.plotly_chart(wordcloud, use_container_width=True)
            right.plotly_chart(wordbar, use_container_width=True)

            # st.write('Topic Modelling output', modelled_data)
            # for values in modelled_data:

            # labelling the read data
            topic_results, recommendations = topic_modelling.generate_topic_labels(modelled_data)
            # st.write(modelled_data)
            # st.write(topic_results.text)
            # st.write('Current topics on Social Media')

            # generate recommendations using Gemini's capabilities and the input data
            # st.write("LLM's Recommendations")
            # for rec in recommendations:
            #     st.write(rec.text)
            #     # st.write(rec.text)

            # *** trial code ***
            # Assuming 'Cases' contains accuracy scores
            # st.write('The result_df', result_df)
            accuracy_scores = result_df['score']

            # Fit GAMLSS model
            mu, sigma, shape = comment_analysis.fit_gamlss(accuracy_scores)
            print('Sigma Value',sigma)

            # Detect anomalies
            # anomalies = comment_analysis.detect_anomalies(accuracy_scores, sigma)
            anomalies, anomaly_records = comment_analysis.detect_anomalies(result_df, sigma)
            st.write('The anomaly records', anomaly_records)

            # Plot accuracy scores with anomalies
            # new_plt = comment_analysis.plot_data_with_anomalies(accuracy_scores, anomalies)
            # new_plt = comment_analysis.plot_data_with_anomalies(result_df, anomalies)

            # latest code to integrate ground truth
            ground_truth_data = pd.read_csv('data/africa_cases')
            new_ground_data = pd.read_csv('data/covid_19_ground_truth.csv')

            # Selecting specific columns from each DataFrame
            selected_cols_df_one = ground_truth_data[['date', 'Confirmed']]
            selected_cols_df_two = new_ground_data[['date', 'Confirmed']]

            # Merging the two DataFrames based on common columns
            ground_truth_data = pd.merge(selected_cols_df_one, selected_cols_df_two, on=['date', 'Confirmed'],
                                         how='inner')
            plt, anomaly_detection_result = comment_analysis.anomaly_detection(result_df)
            sentiment_score_trend = topic_modelling.sentiment_scores_graph(anomaly_detection_result)
            # st.pyplot(sentiment_score_trend)
            scores = anomaly_detection_result['score']
            # st.write('anomaly results', anomaly_detection_result)
            # new_plt = comment_analysis.plot_data_with_anomalies(ground_truth_data, anomaly_records)
            # st.pyplot(new_plt)
            ground_truth = comment_analysis.merge_ground_truth(anomaly_detection_result, ground_truth_data)
            # detected_anomalies = comment_analysis.detect_anomalies(anomaly_detection_result, sigma)
            # detected_anomalies = comment_analysis.detect_anomalies(scores, sigma)
            detected_anomalies = comment_analysis.detect_anomalies(anomaly_detection_result, sigma)
            # st.write('detected anomalies', anomalies)
            # st.write('Ground truth', ground_truth)
            precision, recall, f1_score = comment_analysis.calculate_metrics(ground_truth, anomalies)

            precision_value, recall_value, f1_score_value = st.columns(3, gap='small')
            with precision_value:
                st.info('Precision Score', icon="🔍")
                st.metric(label="Precision", value=f"{precision*100:.01f}%")
            with recall_value:
                st.info('Recall Score', icon="📋")
                st.metric(label="Recall", value=f"{recall*100:.01f}%")
            with f1_score_value:
                st.info('F1_Score', icon="📉")
                st.metric(label="F1 Score", value=f"{f1_score*100:.01f}%")
            style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",
                               box_shadow=True)

            # end of code to integrate ground truth

            # *** end of trial code ***


if __name__ == '__main__':
    main()
