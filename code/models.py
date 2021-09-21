"""
Models for event forecasting. 

Also has the multi-instance learning model. 

Author: Alexandra DeLucia
"""
# Standard imports
import logging
import pickle
import sys

# Third-party imports
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.ensemble 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.base import BaseEstimator
import numpy as np
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class MyCV:
    """Custom split for train / test data to work with scikit-learn CV"""
    def __init__(self, indices, splits=1):
        self.train_indices = indices[0]
        self.test_indices = indices[1]
        self.splits = splits

    def __iter__(self):
        return self

    def split(self, X, y, groups):
        for n in range(self.splits):
            logging.debug(f"In MyCV: On split {n}")
            yield self.train_indices, self.test_indices


class CustomModel(BaseEstimator):
    """
    Extends the scikit-learn BaseEstimator class to have a streamlined interface
    (i.e. "fit", "train") and adds expansions for easily saving the model
    and feature importances.
    """
    def __init__(self):
        return

    def fit(self, X_train, y_train):
        return
    
    def predict(self, X_test):
        return
    
    def predict_proba(self, X_test):
        return
    
    def save(self):
        return

    def get_feature_importance(self):
        return


class RandomThresholdClassifier(CustomModel):
    def __init__(self):
        super().__init__()
        self.pos_rate = None

    def fit(self, X_train, y_train, sample_weight=None):
        self.pos_rate = y_train.sum() / len(y_train)

    def predict(self, X_test):
        predictions = [
            np.random.choice([0, 1], p=[1-self.pos_rate, self.pos_rate]) for _ in range(len(X_test))
        ]
        return np.array(predictions)

    def predict_proba(self, X_test):
        probabilities = [
            [1-self.pos_rate, self.pos_rate] for i in range(len(X_test))
        ]
        return np.array(probabilities)

    def get_feature_importance(self):
        return [0]


class CountryThresholdClassifier(CustomModel):
    def __init__(self, train_df, test_df):
        super().__init__()
        self.country_baseline = (train_df.groupby("COUNTRY").label.sum() / train_df.groupby("COUNTRY").size()).to_dict()
        self.test_idx_lookup = test_df["COUNTRY"].to_dict()
    
    def fit(self, X_train, y_train):
        return

    def predict(self, X_test):
        predictions = []
        for i in X_test.index:
            pos_prob = self.country_baseline[self.test_idx_lookup[i]]
            predictions.append(
                np.random.choice([0,1], p=[1-pos_prob, pos_prob])
            )
        return np.array(predictions)

    def predict_proba(self, X_test):
        probabilities = [
            [1-self.country_baseline[self.test_idx_lookup[i]], self.country_baseline[self.test_idx_lookup[i]]] for i in X_test.index
        ]
        return np.array(probabilities)

    def get_feature_importance(self):
        return [0]


class RandomForestClassifier(CustomModel):
    """
    Wraps scikit-learn random forest
    """
    def __init__(self, scale=False, n_jobs=-1, **kwargs):
        super().__init__()
        # Use pipeline so the data is automatically scaled
        pipeline = []
        if scale:
            pipeline.append(("scaler", StandardScaler()))
        pipeline.append((
            "rf",
            sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_depth=32, min_samples_split=32, n_jobs=n_jobs, class_weight='balanced', **kwargs)
        ))
        self.clf = Pipeline(pipeline)
        return
    
    def fit(self, X_train, y_train, sample_weight=None):
        self.clf.fit(X_train, y_train, rf__sample_weight=sample_weight)

    def predict(self, y_test):
        return self.clf.predict(y_test)

    def predict_proba(self, y_test):
        return self.clf.predict_proba(y_test)

    def get_feature_importance(self):
        return self.clf['rf'].feature_importances_


class SupportVectorMachineClassifier(CustomModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.clf = svm.SVC(gamma=1, probability=True)
    
    def fit(self, X_train, y_train, sample_weight=None):
        self.clf.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, y_test):
        return self.clf.predict(y_test)

    def predict_proba(self, y_test):
        return self.clf.predict_proba(y_test)

    def get_feature_importance(self):
        return self.clf.feature_importances_


class LogisticRegressionClassifier(CustomModel):
    """
    Wraps scikit-learn logistic regression. Contains scaling option.
    """
    def __init__(self, scale=False, **kwargs):
        super().__init__()
        # Logistic regression
        # Use pipeline so the data is automatically scaled
        pipeline = []
        if scale:
            pipeline.append(("scaler", StandardScaler()))
        pipeline.append((
            "lr",
            LogisticRegression(solver="liblinear", max_iter=100, penalty="l1", **kwargs)
        ))
        self.clf = Pipeline(pipeline)
        return

    def fit(self, X_train, y_train, sample_weight=None):
        self.clf.fit(X_train, y_train, lr__sample_weight=sample_weight)

    def predict(self, y_test):
        return self.clf.predict(y_test)

    def predict_proba(self, y_test):
        return self.clf.predict_proba(y_test)
    
    def get_feature_importance(self):
        return self.clf["lr"].coef_

