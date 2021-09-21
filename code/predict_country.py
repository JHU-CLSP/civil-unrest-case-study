'''
Predict country using Tweet data, adapted from event_forecasting.py

Author: Jack Zhang, Alexandra DeLucia
'''
# Standard
import argparse
import logging
import os
import pickle

# Third-party
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np

# Convenience wrappers for consistent interfaces
from models import RandomThresholdClassifier, CountryThresholdClassifier
from models import LogisticRegressionClassifier, RandomForestClassifier, SupportVectorMachineClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_model(model_name, scale=False, n_jobs=None):
    if model_name == "rf":
        return RandomForestClassifier(scale, n_jobs)
    if model_name == "lr":
        return LogisticRegressionClassifier(scale)
    if model_name == "random":
        return RandomThresholdClassifier()
    if model_name == "country-random": 
        return CountryThresholdClassifier()
    if model_name == "svm":
        return SupportVectorMachineClassifier()

def parse_args():
    parser = argparse.ArgumentParser()
    # Data and labels
    parser.add_argument("--features", type=str,  required=True,
                        help="Path to daily features for the countries in TSV format <country><date><features>")
    parser.add_argument("--output-dir", help="Folder to save results", required=True)
    parser.add_argument("--use-existing-model", action="store_true",
                        help="If model file exists, use that model instead of overwriting")

    # Experiment settings
    parser.add_argument("--train-years", type=str, nargs="+", default=["2014", "2015", "2016", "2017"], 
                        help="Years to use in the train set.")
    parser.add_argument("--validation-years", type=str, nargs="+", default=None,
                        help="Years to use in the validation set.")
    parser.add_argument("--test-years", type=str, nargs="+", default=["2018", "2019"],
                        help="Years to use in the test set.")
    parser.add_argument("--n-iter", type=int, default=1, help="Number of repeated model train / test iterations for each task")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of processes for models")
    parser.add_argument("--models", nargs="*", default=["random", "rf", "lr"], choices=["rf", "lr", "random", "country-random", "svm"],
                        help="Model for event forecasting/detection. "
                             "`rf` for random forest and `lr` for logistic regression")
    
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def load_dataset(args):
    """Load the dataset and format it according to commandline options"""
    logging.info("Loading, aggregating, and labelling features")
    
    # Load features
    dataset_df = pd.read_csv(
        args.features,
        header=0,
        index_col=1,  # country, date multi-index
        delimiter="\t",
        parse_dates=[1]
    )
    logging.info(dataset_df.head())

    # Convert any string features to numbers with LabelEncoder
    # This fixes the "ValueError" for strings that sklearn classifiers throw
    encoder = LabelEncoder()
    dataset_df["COUNTRY"] = encoder.fit_transform(dataset_df["COUNTRY"])
    dataset_df.reset_index(inplace=True)
    label_index = 1
    feature_index = 2
    num_features = len(dataset_df.columns) - feature_index
    logging.debug(f"num features: {num_features}")

    return dataset_df, label_index, feature_index, encoder

def dataset_split(dataset_df, args):
    train_df = dataset_df[dataset_df.DATE.map(lambda x: x.strftime("%Y") in args.train_years)]
    if args.validation_years:
        valid_df = dataset_df[dataset_df.DATE.map(lambda x: x.strftime("%Y") in args.validation_years)]
    else:
        valid_df = None
    test_df = dataset_df[dataset_df.DATE.map(lambda x: x.strftime("%Y") in args.test_years)]

    return train_df, valid_df, test_df

def eval_model(train_df, test_df, label_index, feature_index, encoder, clf, n_iter, output_file):
    X_train = train_df.iloc[:, feature_index:]
    y_train = train_df.iloc[:, label_index]
    X_test = test_df.iloc[:, feature_index:]
    y_test = test_df.iloc[:, label_index]

    f1, prec, recall = [], [], []
    for n in range(n_iter):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Save results (positive class)
        prec_temp, recall_temp, f1_temp = precision_recall_fscore_support(y_test, y_pred)[:3]
        logging.debug(f"On iter {n} with f1: {f1_temp[1]}, precision: {prec_temp[1]}, recall: {recall_temp[1]}")
        f1.append(f1_temp[1])
        prec.append(prec_temp[1])
        recall.append(recall_temp[1])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f = open(output_file, "w")
    f.write(f"f1: {np.average(f1):.2f} ({np.std(f1):.3f}), precision: {np.average(prec):.2f} ({np.std(prec):.3f}), recall: {np.average(recall):.2f} ({np.std(recall):.3f})\n")
    f.write(classification_report(y_test, y_pred, target_names=encoder.classes_))
    f.close()

    return clf


def run_experiment(args, train_df, valid_df, label_index, feature_index, encoder):
    # Run experiments
    for model_name in args.models:
        base_file_name = os.path.join(args.output_dir, f"country_prediction_{model_name}")
        results_file = f"{base_file_name}_results.out"
        model_file = f"{base_file_name}_model.pkl"
        if args.use_existing_model and os.path.exists(model_file):
            logging.info(f"Loading model from {model_file}")
            with open(model_file, "rb") as f:
                clf = pickle.load(f)
        else:
            clf = load_model(model_name, n_jobs=args.n_jobs)

        clf = eval_model(train_df, valid_df, label_index, feature_index, encoder, clf, args.n_iter, results_file)

        # Save model
        if not args.use_existing_model:
            pickle.dump(clf, open(model_file, "wb"), protocol=3)
    settings_file = os.path.join(args.output_dir, f"country_prediction_args.pkl")
    pickle.dump(args, open(settings_file, "wb"), protocol=3)


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    dataset_df, label_index, feature_index, encoder = load_dataset(args)
    train_df, valid_df, test_df = dataset_split(dataset_df, args)

    if valid_df is not None:
        run_experiment(args, train_df, valid_df, label_index, feature_index, encoder)
    else:
        run_experiment(args, train_df, test_df, label_index, feature_index, encoder)

