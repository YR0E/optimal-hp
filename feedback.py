import streamlit as st
from streamlit_gsheets import GSheetsConnection
from datetime import datetime
import pandas as pd

st.write("Welcome to Feedback")

# Connect to Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

existing_data = conn.read(worksheet="HP_optima", usecols=list(range(5)), ttl=5)
existing_data = existing_data.dropna(how='all')


SECTION_TYPES = [
    "Concept",
    "Documentation",
    "Calculation",
    "Visualization",
    "Other"
]

FEEDBACK_TYPES = [
    "Question",
    "Suggestion",
    "Request",
    "Error",
    "Bug",
    "Other",
]


with st.form(key="feedback_form"):
    
    submitting_date = datetime.now()
    name = st.text_input(label="Name*")
    section_type = st.selectbox("Section*", options=SECTION_TYPES, index=None)
    feedback_type = st.multiselect("Feedback Type", options=FEEDBACK_TYPES)
    comment = st.text_area(label="Comments")

    # Mark mandatory fields
    st.markdown("**required*")

    submit_button = st.form_submit_button(label="Leave feedback")

    # If the submit button is pressed
    if submit_button:
        # Check if all mandatory fields are filled
        if not name or not section_type:
            st.warning("Ensure all mandatory fields are filled.")
            st.stop()

        else:
            # Create a new row of feedback data
            feedback_data = pd.DataFrame(
                [
                    {
                        "Date": submitting_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "Name": name,
                        "Section": section_type,
                        "CommentType": ", ".join(feedback_type),
                        "Comment": comment,
                    }
                ]
            )

            # Add the new feedback to the existing data
            updated_df = pd.concat([existing_data, feedback_data], ignore_index=True)

            # Update Google Sheets with the new feedback
            conn.update(worksheet="HP_optima", data=updated_df)

            st.success("successfully submitted!")