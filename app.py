import streamlit as st
import pandas as pd
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import utils

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data uploading", "Data characteristics", "Initial model", "Refine model" "Export pipeline"])

st.title('LLM assisted AutoML app')

if page == "Data uploading":
    st.write("""#### Upload your data""")

    data = st.file_uploader(" ", type=['csv'])
    if data is not None:
        try:
            if data.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(data)
            else:
                st.error('Please upload csv file')
                df = None

            if st.session_state.df is not None:
                st.write(st.session_state.df)
                st.write("Shape of the dataset:", st.session_state.df.shape)
                st.write("Summary statistics:", st.session_state.df.describe())

        except Exception as e:
            st.error(f"Error loading file: {e}")

if page == "Data characteristics":
    st.write("""#### Data Characteristics""")
    st.header("Data report")
    st.write(utils.data_report(st.session_state.df))

if page == "Initial model":
    st.write("""#### Refine Model""")

    openai_api_key = '"YOUR API KEY HERE"'
    llm = ChatOpenAI(temperature=0.6, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(return_messages=True)
    system_message = f"""You are a senior data scientist tasked with the guiding of the use of an autoML tool to 
    discover the best model type and model configuration for a classification task on a dataset.Your role involves 
    understanding the dataset characteristics, proposing suitable metrics, hyperparameters, and their search spaces, 
    analyzing results, and iterating on configurations. """

    st.header("GPT Chat")
    user_input = st.text_area("Enter your prompt here:", "")
