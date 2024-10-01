import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download stopwords if not already available
nltk.download('punkt')
nltk.download('stopwords')

st.title("Data Analysis")

@st.cache_data
def load_data(file):
    data = pd.read_excel(file)
    return data

with st.sidebar:
    st.header("Configuration")
    upload = st.file_uploader("Choose a file")
    
    if upload is not None:
        df = load_data(upload)
        
        # Predefined filter options
        predefined_tactics = ["Defense Evasion", "Privilege Escalation", "Initial Access", "Execution", "Persistence"]
        predefined_platforms = ["Windows", "Linux", "macOS", "AWS", "GCP", "Azure"]
        predefined_data_source = ["Network Traffic", "Process Logs", "File Monitoring", "Command-line Activity"]
        predefined_permissions = ["Administrator", "User", "root", "System"]

        st.subheader("Filters")
        name_filter = st.text_input("Name contains", "")
        # Predefined Tactics filter
        tactics_filter = st.multiselect("Tactics", options=predefined_tactics)
        # Predefined Platforms filter
        platforms_filter = st.multiselect("Platforms", options=predefined_platforms)
        # Predefined Data Source filter
        data_source_filter = st.multiselect("Data Source", options=predefined_data_source)
        # Predefined Permissions Required filter
        permissions_filter = st.multiselect("Permissions Required", options=predefined_permissions)

if upload is None:
    st.info("Upload a file through the sidebar.")
    st.stop()

df_clean = df.replace(np.nan, 0)

# Apply filters
df_filtered = df.copy()

if name_filter:
    df_filtered = df_filtered[df_filtered["name"].str.contains(name_filter, case=False)]

if tactics_filter:
    df_filtered = df_filtered[df_filtered["tactics"].isin(tactics_filter)]

if platforms_filter:
    df_filtered = df_filtered[df_filtered["platforms"].isin(platforms_filter)]

if data_source_filter:
    df_filtered = df_filtered[df_filtered["data source"].isin(data_source_filter)]

if permissions_filter:
    df_filtered = df_filtered[df_filtered["permissions required"].isin(permissions_filter)]

with st.expander("Data Preview"):
    st.dataframe(df_filtered)

# Create a bar graph of the tactics used
tactics_count = df_filtered["tactics"].value_counts().reset_index()
tactics_count.columns = ["Tactic", "Count"]

fig = px.bar(tactics_count, x="Tactic", y="Count", title="Tactics Used", labels={"Tactic": "Tactic", "Count": "Number of Occurrences"})
st.plotly_chart(fig)

# Pie chart 
fig = px.pie(tactics_count, values="Count", names="Tactic", title="Tactics Distribution", labels={"Tactic": "Tactic", "Count": "Number of Occurrences"})
st.plotly_chart(fig)


# TF-IDF Processing
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in set(stopwords.words('english'))]
    return ' '.join(filtered_words)

# Apply stopwords removal to each document in the 'name' column
df_filtered['name_no_stopwords'] = df_filtered['name'].apply(remove_stopwords)

# Vectorize using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtered['name_no_stopwords'])

# Convert TF-IDF matrix to a DataFrame for better display in Streamlit
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Aggregate TF-IDF scores
tfidf_totals = np.array(tfidf_matrix.sum(axis=0)).flatten()
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create a dictionary for word frequencies based on TF-IDF scores
word_weights = {feature_names[i]: tfidf_totals[i] for i in range(len(feature_names))}

# Generate the WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_weights)

# Display the WordCloud using Matplotlib
st.subheader("TF-IDF Word Cloud")
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis('off')
st.pyplot(fig)
