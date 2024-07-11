import torch
import numpy as np
import pickle as pkl
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve

def diabetes_data(knn=5):
    cwd=os.getcwd()
    with open(f"{cwd}/data/diabetes/X.pkl", "rb") as file:
        X_raw = pkl.load(file)
    with open(f"{cwd}/data/diabetes/y.pkl", "rb") as file:
        y = pkl.load(file)

    y_counts = np.unique(y, return_counts=True)[1]
    weight = torch.tensor([y_counts[0]/y_counts[1]], dtype=torch.float32)

    imputer = KNNImputer(n_neighbors=knn)
    X_imputed_not_norm = imputer.fit_transform(X_raw)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_imputed_not_norm)

    return X, y, weight

def sig(x):
  return 1 / (1 + np.exp(-x))

def fmax_score(y_test, y_score, beta=1.0, pos_label=1):
    fmax_score, _, _, threshold_fmax = fmax_precision_recall_threshold(
        y_test, y_score, beta=beta, pos_label=pos_label
    )
    return fmax_score, threshold_fmax


def fmax_precision_recall_threshold(labels, y_score, beta=1.0, pos_label=1):
    """
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein
    Function Prediction. Nature Methods, 10(3), 221-227.
    Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In
    Introduction to Information Retrieval. Cambridge University Press.
    """
    if pos_label == 0:
        labels = 1 - np.array(labels)
        y_score = 1 - np.array(y_score)

    precision_scores, recall_scores, thresholds = precision_recall_curve(
        labels, y_score
    )

    np.seterr(divide="ignore", invalid="ignore")
    f_scores = (
        (1 + beta**2)
        * (precision_scores * recall_scores)
        / ((beta**2 * precision_scores) + recall_scores)
    )

    arg_fmax = np.nanargmax(f_scores)

    fmax_score = f_scores[arg_fmax]
    precision_fmax = precision_scores[arg_fmax]
    recall_fmax = recall_scores[arg_fmax]
    threshold_fmax = thresholds[arg_fmax]

    return fmax_score, precision_fmax, recall_fmax, threshold_fmax