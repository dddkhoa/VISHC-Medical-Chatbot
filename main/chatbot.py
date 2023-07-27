import os

import langchain
import streamlit as st
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

from main import config

langchain.verbose = False

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY


class Chatbot:
    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    qa_template = """
        You are a helpful AI assistant. Use the following pieces of context to answer user's question.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Use as much detail as possible when responding.

        context: {context}
        =========
        question: {question}
        ======
        """

    QA_PROMPT = PromptTemplate(
        template=qa_template, input_variables=["context", "question"]
    )

    def chat_with_history(self, query):
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
            max_tokens_limit=4097,
            combine_docs_chain_kwargs={"prompt": self.QA_PROMPT},
        )

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]

    def chat(self, query):
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.vectors.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.QA_PROMPT},
        )

        result = chain({"query": query})
        return result["result"]
