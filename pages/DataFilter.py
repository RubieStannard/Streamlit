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

st.title("MITRE ATT&CK Data Analysis")

@st.cache_data
def load_data(file):
    data = pd.read_excel(file)
    return data

# Load file
with st.sidebar:
    st.header("Configuration")
    upload = st.file_uploader("Choose a file")

# Store button states for each filter category
selected_filters = {
    "tactics": False,
    "platforms": False,
    "data_sources": False,
    "defenses_bypassed": False  # New category for Defenses Bypassed
}

if upload is not None:
    df = load_data(upload)
    
    st.subheader("Filters")
    
    # Tactics filter button
    if st.button("Tactics"):
        selected_filters["tactics"] = not selected_filters["tactics"]
        
    # Platforms filter button
    if st.button("Platforms"):
        selected_filters["platforms"] = not selected_filters["platforms"]
        
    # Data Sources filter button
    if st.button("Data Sources"):
        selected_filters["data_sources"] = not selected_filters["data_sources"]
        
    # Defenses Bypassed filter button
    if st.button("Defenses Bypassed"):
        selected_filters["defenses_bypassed"] = not selected_filters["defenses_bypassed"]

    # Logic to apply filters based on selected categories
    selected_categories = [key for key, val in selected_filters.items() if val]
    
    if len(selected_categories) > 0:
        st.write(f"Selected filters: {', '.join(selected_categories)}")
        
        df_filtered = df.copy()
        
        # Automatically filter and display data based on selected categories
        if "tactics" in selected_categories:
            st.subheader("Tactics Data")
            tactics_data = df_filtered[["tactics"]]  # Assuming the column name is 'tactics'
            st.dataframe(tactics_data)
            
            # Bar chart for tactics data
            tactics_count = tactics_data["tactics"].value_counts().reset_index()
            tactics_count.columns = ["Tactic", "Count"]
            fig = px.bar(tactics_count, x="Tactic", y="Count", title="Tactics Used", labels={"Tactic": "Tactic", "Count": "Number of Occurrences"})
            st.plotly_chart(fig)
        
        if "platforms" in selected_categories:
            st.subheader("Platforms Data")
            platforms_data = df_filtered[["platforms"]]  # Assuming the column name is 'platforms'
            st.dataframe(platforms_data)
            
            # Bar chart for platforms data
            platforms_count = platforms_data["platforms"].value_counts().reset_index()
            platforms_count.columns = ["Platform", "Count"]
            fig = px.bar(platforms_count, x="Platform", y="Count", title="Platforms Used", labels={"Platform": "Platform", "Count": "Number of Occurrences"})
            st.plotly_chart(fig)
        
        if "data_sources" in selected_categories:
            st.subheader("Data Sources Data")
            data_sources_data = df_filtered[["data_sources"]]  # Assuming the column name is 'data_sources'
            st.dataframe(data_sources_data)
            
            # Pie chart for data sources distribution
            data_sources_count = data_sources_data["data_sources"].value_counts().reset_index()
            data_sources_count.columns = ["Data Source", "Count"]
            fig = px.pie(data_sources_count, values="Count", names="Data Source", title="Data Sources Distribution", labels={"Data Source": "Data Source", "Count": "Number of Occurrences"})
            st.plotly_chart(fig)
        
        if "defenses_bypassed" in selected_categories:
            st.subheader("Defenses Bypassed Data")
            defenses_bypassed_data = df_filtered[["defenses_bypassed"]]  # Assuming the column name is 'defenses_bypassed'
            st.dataframe(defenses_bypassed_data)
            
            # Bar chart for defenses bypassed
            defenses_bypassed_count = defenses_bypassed_data["defenses_bypassed"].value_counts().reset_index()
            defenses_bypassed_count.columns = ["Defense Bypassed", "Count"]
            fig = px.bar(defenses_bypassed_count, x="Defense Bypassed", y="Count", title="Defenses Bypassed", labels={"Defense Bypassed": "Defense Bypassed", "Count": "Number of Occurrences"})
            st.plotly_chart(fig)
        
        # TF-IDF Processing and WordCloud generation for the name column
        def remove_stopwords(text):
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.lower() not in set(stopwords.words('english'))]
            return ' '.join(filtered_words)
        
        df_filtered['name_no_stopwords'] = df_filtered['name'].apply(remove_stopwords)
        
        # Vectorize using TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtered['name_no_stopwords'])
        
        # Convert TF-IDF matrix to DataFrame for better display
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

else:
    st.info("Upload a file through the sidebar.")
