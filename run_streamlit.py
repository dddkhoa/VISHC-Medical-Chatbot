import re
import sys
from io import StringIO

import streamlit as st

from main.history import ChatHistory
from main.pdf_handler import PDFHandler
from main.sidebar import Sidebar
from main.utils import Utils


def prompt_form():
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area(
            "Query:",
            placeholder="Ask me anything about the document...",
            key="input",
            label_visibility="collapsed",
        )
        submit_button = st.form_submit_button(label="Send")

        is_ready = submit_button and user_input
    return is_ready, user_input


def main():
    st.set_page_config(layout="wide", page_icon="💬", page_title="Medical ChatBot 🤖")

    history = ChatHistory()
    sidebar = Sidebar()
    utils = Utils()
    sidebar.show_options()

    try:
        chatbot = utils.setup_chatbot()
        st.session_state["ready"] = True
        st.session_state["chatbot"] = chatbot
    except Exception as e:
        print(e)
        st.error(f"Error: {str(e)}")

    try:
        if st.session_state["ready"]:
            pdf = st.file_uploader(
                "Upload a Medical PDF file (in either EN or VI)", type=["pdf"]
            )

            if pdf:
                st.session_state["uploaded_file_name"] = pdf.name
                pdf_handler = PDFHandler(pdf)
                pdf_handler.upload_to_weaviate()

                response_container, prompt_container = st.container(), st.container()

                with prompt_container:
                    is_ready, user_input = prompt_form()

                    history.initialize()

                    if st.session_state["reset_chat"]:
                        history.reset()

                    if is_ready:
                        history.append("user", user_input)

                        old_stdout = sys.stdout
                        sys.stdout = captured_output = StringIO()

                        output = st.session_state["chatbot"].chat_with_history(
                            user_input
                        )

                        sys.stdout = old_stdout

                        history.append("assistant", output)

                        # Clean up the agent's thoughts to remove unwanted characters
                        thoughts = captured_output.getvalue()
                        cleaned_thoughts = re.sub(
                            r"\x1b\[[0-9;]*[a-zA-Z]", "", thoughts
                        )
                        cleaned_thoughts = re.sub(r"\[1m>", "", cleaned_thoughts)

                        with st.expander("Display the agent's thoughts"):
                            st.write(cleaned_thoughts)

                history.generate_messages(response_container)

            else:
                st.warning("Please upload a PDF file to start the chatbot.")

    except Exception as e:
        print(e)
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
