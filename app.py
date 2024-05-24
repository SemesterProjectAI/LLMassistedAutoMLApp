import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
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
page = st.sidebar.radio("Go to", ["Data uploading", "Data characteristics", "Metrics suggestions", "Initial model", "Refine model" "Export pipeline"])

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
    st.session_state.report, st.session_state.target = utils.data_report(st.session_state.df)
    st.write(st.session_state.report)

if page == "Metrics suggestions":
    st.write("""#### Metrics suggestions""")
    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(st.session_state.df, st.session_state.target, test_size=0.2, random_state=42)
    openai_api_key = '"YOUR_OPENAI_API_KEY"'
    llm = ChatOpenAI(temperature=0.6, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(return_messages=True)
    st.session_state.last_run_best_score = []
    st.session_state.all_time_best_score = []
    system_message = f"""
    You are a senior data scientist tasked with guiding the use of an AutoML tool  
    to discover the best XGBoost model configurations for a given binary classification dataset. 
    Your role involves understanding the dataset characteristics, proposing suitable metrics, 
    hyperparameters, and their search spaces, analyzing results, and iterating on configurations. 
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("""{input}""")
    ])
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=False)

    prompt = utils.suggest_metrics(st.session_state.report)
    response = conversation.predict(input=prompt)
    st.header("GPT Chat")
    st.write(response)
