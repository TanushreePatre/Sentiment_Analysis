import streamlit as st
import pandas as pd
import re
from transformers import pipeline
import ast
import os

# Initialize the sentiment analysis pipeline
sent_pipeline = pipeline("sentiment-analysis")

# Function to preprocess ratings
def preprocess_ratings(remark):
    # Convert remark to string if it's not already
    remark = str(remark)
    match = re.findall(r'\d+', remark)
    if match:
        value = max(map(int, match))
        if value > 5:
            return "positive"
        else:
            return "negative"
    else:
        return remark

# Function to extract label and score
def extract_label_and_score(review):
    try:
        review_dict = ast.literal_eval(str(review))
        return pd.Series([review_dict['label'], review_dict['score']])
    except (ValueError, KeyError):
        return pd.Series([None, None])

# Function to perform sentiment analysis and save the output
def sentiment_analysis_pipeline(input_file, output_file_path):

    data = pd.read_csv(input_file)

    data['Processed_Remark'] = data['Remarks'].apply(preprocess_ratings)

    data['Reviews'] = data['Processed_Remark'].apply(lambda x: sent_pipeline(x)[0])

    data[['label', 'score']] = data['Reviews'].apply(extract_label_and_score)

    data.drop(columns=['Reviews', 'Processed_Remark'], inplace=True)

    data.to_csv(output_file_path, index=False)

    return output_file_path

# Streamlit app layout
st.title("Sentiment Analysis App")

# File upload block
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    input_file_path = "new_data.csv"
    with open(input_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform sentiment analysis
    output_file_path = "output.csv"
    sentiment_analysis_pipeline(input_file_path, output_file_path)

    st.success("Sentiment analysis completed!")

    # Provide download button for the output file
    with open(output_file_path, "rb") as f:
        st.download_button(
            label="Download Processed File",
            data=f,
            file_name="output_data.csv",
            mime="text/csv"
        )
    
    # Clean up temporary files
    os.remove(input_file_path)
    os.remove(output_file_path)
