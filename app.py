import streamlit as st
import pandas as pd
import ast
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
import re
import utils
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint, uniform

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data uploading", "Data characteristics", "Metrics suggestions", "Initial model", "Refine model"])

st.title('LLM assisted AutoML app')
openai_api_key = '"YOUR API KEY HERE"'
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
                st.session_state.user_input = st.text_area("Please, briefly describe your data:", "")

        except Exception as e:
            st.error(f"Error loading file: {e}")

if page == "Data characteristics":
    st.header("Data Characteristics")
    st.write("""#### Data Report""")
    st.session_state.df, st.session_state.report, st.session_state.target = utils.data_report(st.session_state.df)
    st.write(st.session_state.report)

if page == "Metrics suggestions":
    st.header("Metrics suggestions")

    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(st.session_state.df, st.session_state.target, test_size=0.2, random_state=42)

    prompt = utils.suggest_metrics(st.session_state.user_input, st.session_state.report)
    response = conversation.predict(input=prompt)
    st.write("""#### GPT Chat""")
    st.write(response)
    st.session_state.metric = utils.extract_metric(response)

if page == "Initial model":

    st.header("Initial Model")
    prompt = utils.suggest_initial_search_space()
    response = conversation.predict(input=prompt)
    st.write(response)
    search_space_string = utils.extract_search_space(response)
    search_space_dict = {}
    if search_space_string:
        import scipy.stats as stats
        from scipy.stats import randint, uniform
        exec(search_space_string, globals(), search_space_dict)
        search_space = search_space_dict.get('search_space', {})
        clf = xgb.XGBClassifier(seed=42, objective='binary:logistic', enable_categorical=True, eval_metric=st.session_state.metric, use_label_encoder=False)

        search = HalvingRandomSearchCV(clf, search_space, scoring=st.session_state.metric, n_candidates=50,
                                       cv=5, min_resources='exhaust', factor=3, verbose=1).fit(st.session_state.X_train, st.session_state.y_train)

        y_pred = search.predict(st.session_state.X_test)
        y_pred_proba = search.predict_proba(st.session_state.X_test)
        utils.metrics_display(st.session_state.y_test, y_pred, y_pred_proba[:, 1])
    else:
        st.error("No search space found for initial model")


if page == "Refine model":
    st.header("Refine Model")
