import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef
)


def metrics_display(y_test, y_pred, y_pred_proba):
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f'ROC_AUC score: {roc_auc_score(y_test, y_pred_proba):.3f}')
    print(f'f1 score: {f1_score(y_test, y_pred):.3f}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
    print(f'Precision: {precision_score(y_test, y_pred) * 100:.2f}%')
    print(f'Detection rate: {recall_score(y_test, y_pred) * 100:.2f}%')
    print(f'False alarm rate: {fp / (tn + fp) * 100}%')
    print(f'MCC: {matthews_corrcoef(y_test, y_pred):.2f}')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


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

    class_counts = target.value_counts()
    class_distribution = class_counts / num_instances
    if any(class_distribution < 0.3) or any(class_distribution > 0.7):
        class_imbalance = True
    else:
        class_imbalance = False

    report = f"""Data characteristics report :
        - General information :
            - Number of instances : {num_instances}
            - Number of features : {num_features}
        
        - Class Distribution analysis :
            - Class distribution : {class_distribution.to_string()}
        {'Warning : Class imbalance detected' if class_imbalance else 'No class imbalance detected'}
    
        - Feature analysis :
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
    return report, target


def suggest_metrics(report):
    prompt = f"""
    The classification problem under investigation is based on a network intrusion detection dataset. 
    This dataset contains DOS, Probe, R2L, and U2R attack types, which are all grouped under the 
    "attack" class (label: 1). Conversely, the "normal" class is represented by label 0. 
    Below are the dataset's characteristics:
    {report}.

    For this specific inquiry, you are tasked with recommending a suitable hyperparameter optimization 
    metric for training a XGBoost model. It is crucial that the model should accurately identify genuine 
    threats (attacks) without raising excessive false alarms on benign activities. They are equally important.
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
