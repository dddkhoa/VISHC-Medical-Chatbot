import streamlit as st


class Sidebar:
    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    @staticmethod
    def show_meta_data():
        st.info(
            """
            Thank you for your interest in our application.
            Please be aware that this is only a Proof of Concept system
            and may contain bugs or unfinished features.
            """
        )

    def show_options(self):
        with st.sidebar:
            self.show_meta_data()
        with st.sidebar.expander("Configuration", expanded=False):
            self.reset_chat_button()
