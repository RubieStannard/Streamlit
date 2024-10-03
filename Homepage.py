import streamlit as st
from DataFilter import load_data, filter_by_category

# Set the page layout and title
st.set_page_config(page_title="MITRE ATT&CK Interactive Dashboard", layout="wide")

# Title for the homepage
st.title("MITRE ATT&CK Data Analysis App")

# Sidebar for file upload
st.sidebar.title("Navigation")
uploaded_file = st.sidebar.file_uploader("Upload the MITRE ATT&CK dataset (.xlsx)", type=["xlsx"])

# Check if the user has uploaded a file
if uploaded_file:
    # Load the data from the uploaded file
    df = load_data(uploaded_file)
    
    # Sidebar filter buttons
    st.sidebar.subheader("Filter Options")
    view_tactics = st.sidebar.button("View Tactics Data")
    view_platforms = st.sidebar.button("View Platforms Data")
    view_data_sources = st.sidebar.button("View Data Sources")
    view_defenses_bypassed = st.sidebar.button("View Defenses Bypassed Data")

    # Display filtered data based on user selection
    if view_tactics:
        st.subheader("Tactics Data")
        filtered_data = filter_by_category(df, "tactics")
        st.dataframe(filtered_data)

    if view_platforms:
        st.subheader("Platforms Data")
        filtered_data = filter_by_category(df, "platforms")
        st.dataframe(filtered_data)

    if view_data_sources:
        st.subheader("Data Sources")
        filtered_data = filter_by_category(df, "data_sources")
        st.dataframe(filtered_data)

    if view_defenses_bypassed:
        st.subheader("Defenses Bypassed Data")
        filtered_data = filter_by_category(df, "defenses_bypassed")
        st.dataframe(filtered_data)
else:
    # Inform the user to upload a file if they haven't yet
    st.info("Please upload a file to proceed.")

# Add any additional logic for data interaction, visualizations, or further analysis here
