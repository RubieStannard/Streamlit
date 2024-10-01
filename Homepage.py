import streamlit as st

# Configure the layout and title
st.set_page_config(page_title="MITRE ATT&CK", layout="wide")

# Title of the homepage
st.title("MITRE ATT&CK DATA")

# Add a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Home", "Data Filter"])

# Load the selected page
if page == "Home":
    st.write("Welcome to the MITRE ATT&CK Data Analysis app!")
    st.write("Use the navigation menu to explore the data filter functionality.")
    # You can add any additional content for the homepage here

elif page == "Data Filter":
    import DataFilter
