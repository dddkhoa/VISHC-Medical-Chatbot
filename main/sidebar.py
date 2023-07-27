import streamlit as st


class Sidebar:
    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    def show_options(self):
        with st.sidebar.expander("Configuration", expanded=False):
            self.reset_chat_button()
