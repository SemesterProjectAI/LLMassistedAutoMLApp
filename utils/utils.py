import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef
)
import re


def metrics_display(y_test, y_pred, y_pred_proba):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    st.write(f'ROC_AUC score: {roc_auc_score(y_test, y_pred_proba):.3f}')
    st.write(f'f1 score: {f1_score(y_test, y_pred):.3f}')
    st.write(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
    st.write(f'Precision: {precision_score(y_test, y_pred) * 100:.2f}%')
    st.write(f'True positive rate: {recall_score(y_test, y_pred) * 100:.2f}%')
    st.write(f'False positive rate: {fp / (tn + fp) * 100}%')
    st.write(f'MCC: {matthews_corrcoef(y_test, y_pred):.2f}')


def data_report(df):
    target = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    num_instances = len(df)
    num_features = features.shape[1]
    int_counts = 0
    binary_counts = 0
    float_counts = 0
    str_counts = 0
    int_list = []
    binary_list = []
    float_list = []
    str_list = []

    for feature in features:
        if df[feature].apply(lambda x: x in [0, 1]).all():
            binary_counts += 1
            binary_list.append(feature)
        elif df[feature].apply(lambda x: isinstance(x, int)).any():
            int_counts += 1
            int_list.append(feature)
        elif df[feature].apply(lambda x: isinstance(x, float)).any():
            float_counts += 1
            float_list.append(feature)
        elif df[feature].apply(lambda x: isinstance(x, str)).any():
            str_counts += 1
            str_list.append(feature)

    for col in str_list:
        df[col] = df[col].astype('category')

    class_counts = target.value_counts()
    class_distribution = class_counts / num_instances
    if any(class_distribution < 0.3) or any(class_distribution > 0.7):
        class_imbalance = True
    else:
        class_imbalance = False
    df_without_target = df.drop(df.columns[-1], axis=1)
    report = f"""
    Data characteristics report :

        General information :
        - Number of instances : {num_instances}
        - Number of features : {num_features}

    Class Distribution analysis :
        Class distribution : {class_distribution.to_string()}\n{'Class imbalance detected' if class_imbalance else 'No class imbalance detected'}

        Feature analysis :
            - Feature names : {features.columns.to_list()}
            - Number of binary features : {binary_counts}
            - Binary features : {binary_list}
            - Number of float features : {float_counts}
            - Float features : {float_list}
            - Number of string features : {str_counts}
            - String features : {str_list}
            - Number of integer features : {int_counts}
            - Integer features : {int_list}
    """
    return df_without_target, report, target


def suggest_metrics(description, report):
    prompt = f"""
    Here is a brief description of the dataset :
    {description}
    Below are the dataset's characteristics:
    {report}.

    For this specific inquiry, you are tasked with recommending a suitable hyperparameter optimization 
    metric for training a XGBoost model.
    Given the problem context and dataset characteristics, suggest only the name of one of the built-in 
    metrics: 
    - 'accuracy'
    - 'roc_auc' (ROCAUC score)
    - 'f1' (F1 score)
    - 'balanced_accuracy' (It is the macro-average of recall scores per class or, equivalently, raw 
    accuracy where each sample is weighted according to the inverse prevalence of its true class) 
    - 'average_precision'
    - 'precision'
    - 'recall'
    - 'neg_brier_score'


    Please first briefly explain your reasoning and then provide the recommended metric name. 
    Your recommendation should be enclosed between markers [BEGIN] and [END], with standalone string for 
    indicating the metric name.
    Do not provide other settings or configurations.
    """

    return prompt


def suggest_initial_search_space():
    prompt = f"""
    Given your understanding of XGBoost and general best practices in machine learning, suggest an 
    initial search space for hyperparameters. 

    Tunable hyperparameters include:
    - n_estimators (integer): Number of boosting rounds or trees to be trained.
    - max_depth (integer): Maximum tree depth for base learners.
    - min_child_weight (integer or float): Minimum sum of instance weight (hessian) needed in a leaf node. 
    - gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree.
    - scale_pos_weight (float): Balancing of positive and negative weights.
    - learning_rate (float): Step size shrinkage used during each boosting round to prevent overfitting. 
    - subsample (float): Fraction of the training data sampled to train each tree. 
    - colsample_bylevel (float): Fraction of features that can be randomly sampled for building each level (or depth) of the tree.
    - colsample_bytree (float): Fraction of features that can be randomly sampled for building each tree. 
    - reg_alpha (float): L1 regularization term on weights. 
    - reg_lambda (float): L2 regularization term on weights. 

    The search space is defined as a dict with keys being hyperparameter names, and values 
    are the search space associated with the hyperparameter. For example:
        search_space = {{
            "learning_rate": loguniform(1e-4, 1e-3)
        }}

    Available types of domains include: 
    - scipy.stats.uniform(loc, scale), it samples values uniformly between loc and loc + scale.
    - scipy.stats.loguniform(a, b), it samples values between a and b in a logarithmic scale.
    - scipy.stats.randint(low, high), it samples integers uniformly between low (inclusive) and high (exclusive).
    - a list of possible discrete value, e.g., ["a", "b", "c"]

    Please first briefly explain your reasoning, then provide the configurations of the initial 
    search space. Enclose your suggested configurations between markers 
    [BEGIN] and [END], and assign your configuration to a variable named search_space.
    """

    return prompt


def extract_search_space(text):
    pattern = re.compile(r'search_space\s*=\s*{[^}]*}', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(0)
    return None


def extract_metric(text):
    pattern = re.compile(r'\[BEGIN] *\'(.*?)\' *\[END]')
    match = pattern.search(text)
    if match:
        return match.group(0)
    return None
