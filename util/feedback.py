import streamlit as st
import pytz
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials


# Define constants
# SECTION_TYPES = ["Concept", "Documentation", "Calculation", "Visualization", "Other"]
FEEDBACK_TYPES = ["Question", "Suggestion", "Feature Request", "Report a Problem", "Other"]
TIMEZONE = pytz.timezone('Asia/Almaty')

# Define the scope for Google API
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]



@st.cache_resource
def get_google_sheet():
    """Initialize and cache the Google Sheets client and worksheet."""
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    client = gspread.authorize(credentials)
    return client.open("Feedback").sheet1  # Access the first sheet


def validate_and_submit_feedback(sheet, feedback_data):
    # Validate the feedback form inputs and submit data to Google Sheets.

    name = feedback_data[1]
    type = feedback_data[3]
    if not name or not type:
        st.warning("Please fill out all required fields.")
    else:
        
        with st.spinner("Sending feedback..."):
            try:
                sheet.append_rows([feedback_data])
                st.success("Feedback sent successfully!")
            except Exception as e:
                st.error(f"Error updating Google Sheets: {e}")


def feedback_form():
    st.markdown("## Feedback")
    with st.expander("Drop a line here", expanded=True):
        with st.form(key="feedback_form", clear_on_submit=True, border=False):
            # Retrieve the cached Google Sheet
            sheet = get_google_sheet()

            # Collecting feedback inputs
            submitting_date = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
            name = st.text_input(label="Name* (*required*)")
            # col1, col2 = st.columns(2)
            # section_type = col1.selectbox("Section*", options=SECTION_TYPES, index=None)
            section_type = 'General'
            feedback_type = st.selectbox("Feedback Type*", options=FEEDBACK_TYPES, index=None)
            comment = st.text_area(label="Comments")
            feedback_data = [submitting_date, name, section_type, feedback_type, comment]

            submit_button = st.form_submit_button(label="Send")
            if submit_button:
                validate_and_submit_feedback(sheet, feedback_data)