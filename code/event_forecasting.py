"""
Forecast events based on daily or weekly features

Author: Alexandra DeLucia, Jack Zhang
"""
# Standard imports
import logging
import warnings
import time
import argparse
import pickle
import datetime as dt
import os
import sys

# Third-party imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit

from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np

# Convenience wrappers for consistent interfaces
from models import MyCV, RandomThresholdClassifier, CountryThresholdClassifier
from models import LogisticRegressionClassifier
from models import RandomForestClassifier
from models import SupportVectorMachineClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    # Data and labels
    parser.add_argument("--features", type=str,  required=True,
                        help="Path to daily features for the countries in TSV format <country>\t<date>\t<features>")
    parser.add_argument("--keep-feature-names", action="store_true",
                        help="Uses TSV header names for features")
    parser.add_argument("--acled-event-data", default="/home/aadelucia/files/minerva/data/acled_daily_events_by_country.csv")
    parser.add_argument("--positive-day-threshold", default=1, type=int,
                        help="Minimum number of events that need to occur on a day in order to be considered a positive sample.")
    parser.add_argument("--output-dir", help="Folder to save results", required=True)
    parser.add_argument("--save-df", action="store_true",
                        help="Saves DataFrame in pickle format. Saves a DataFrame for each task labeling and a '.save' for easy loading")
    parser.add_argument("--from-save", action="store_true",
                        help="Whether to use the aggregated daily DF from a previous run. If True, it checks if "
                             "{output_dir}/{'weekly' if agg_weekly else 'daily'}_{lead_time}_{positive_day_threshold}_df.pkl.save exists."
                             "If this file doesn't exist then no file is used.")
    parser.add_argument("--use-existing-model", action="store_true",
                        help="If model file exists, use that model instead of overwriting")

    # Experiment settings
    parser.add_argument("--time-variance-test", action="store_true", help="Train model on train_years, test on every year after the train_years to experiment the effect of time variance")
    parser.add_argument("--train-years", type=str, nargs="+", default=["2014", "2015", "2016", "2017"], 
                        help="Years to use in the train set.")
    parser.add_argument("--validation-years", type=str, nargs="+", default=None,
                        help="Years to use in the validation set.")
    parser.add_argument("--test-years", type=str, nargs="+", default=["2018", "2019"],
                        help="Years to use in the test set.")
    parser.add_argument("--split-countries", action="store_true",
                        help="NOT IMPLEMENTED. Train and test a model for each country instead of aggregated.")
    parser.add_argument("--agg-weekly", action="store_true", help="Aggregate daily data into weekly")
    parser.add_argument("--lead-time", type=int, default=0,
                        help="NOT IMPLEMENTED. Lead time for prediction. If --agg-weekly then lead-time is calculated in weeks")
    parser.add_argument("--n-iter", type=int, default=1, help="Number of repeated model train / test iterations for each task")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of processes for models")
    parser.add_argument("--models", nargs="*", default=["random", "rf", "lr"], choices=["rf", "lr", "random", "country-random", "svm"],
                        help="Model for event forecasting/detection. "
                             "`rf` for random forest and `lr` for logistic regression")
    parser.add_argument("--run-cv-train", action="store_true", help="Run cross-validation")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--single-country", type=str, nargs="+", default=None, help="Train model on a single country, or a list of countries. Entry 3-letter country code e.g. ZAF")
    parser.add_argument("--exclude-country", type=str, nargs="+", default=None, help="Exclude the list of countries in training. Entry 3-letter country code e.g. ZAF")
    parser.add_argument("--violent-demonstration-only", action="store_true", help="Only consider acled entries where 'sub_event_type' == 'Violent demonstration'")
    parser.add_argument("--weight-by-country", action="store_true", help="Weight samples so that total weight for all instances in a country is equal")
    parser.add_argument("--balance-country-samples", action="store_true", help="Create duplicate data points for countries with very few samples")
    parser.add_argument("--test-on-training", action="store_true", help="evaluate on training set instead of validation set")
    parser.add_argument("--test", action="store_true", help="evaluate on test set instead of validation set")
    parser.add_argument("--fit-scaler", action="store_true", help="fit scaler before classifier to normalize feature")
    
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


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


def run_cv(train_df, feat_index, clf, random_seed, valid_df=None, n_iter=1, n_jobs=1, sample_weights=None):
    X = train_df.iloc[:, feat_index:]
    y = train_df.LABEL

    # Get cross validation scores
    results = cross_validate(
        clf,
        X=X,
        y=y,
        cv=StratifiedShuffleSplit(n_splits=n_iter, test_size=0.1, random_state=random_seed),
        scoring=["f1", "precision", "recall"],
        verbose=0,
        return_estimator=True,
        n_jobs=n_jobs
    )
    feat_importance = np.array([c.get_feature_importance() for c in results["estimator"]])
    formatted_results = {
        "f1": {
            "avg": np.average(results["test_f1"]),
            "std": np.std(results["test_f1"])
        },
        "precision": {
            "avg": np.average(results["test_precision"]),
            "std": np.std(results["test_precision"])
        },
        "recall": {
            "avg": np.average(results["test_recall"]),
            "std": np.std(results["test_recall"])
        },
        "feat_importance": {
            "avg": np.average(feat_importance, axis=0),
            "std": np.std(feat_importance, axis=0)
        }
    }
    # Save final model trained on all training data
    # Evaluate on validation data
    clf.fit(X, y, sample_weight=sample_weights)
    if valid_df:
        X = valid_df.iloc[:, feat_index:]
        y = valid_df.LABEL
        pred_y = clf.predict(X)
        prec, recall, f1 = precision_recall_fscore_support(y, pred_y)[:3]
        formatted_results["validation_score"] = {
            "f1": f1[1],
            "precision": prec[1],
            "recall": recall[1],
            "predictions": pred_y,
            "ground_truth": y
        }
    return formatted_results, clf


def get_country_scores(test_df, feat_index, clf, country_scores):
    for country, temp in test_df.groupby("COUNTRY"):
        X_test = temp.iloc[:, feat_index:]
        y_test = temp.LABEL
        prec, recall, f1 = precision_recall_fscore_support(
            y_test,
            clf.predict(X_test), 
            zero_division=0)[:3]
        if len(prec) < 2:
            logging.debug(f"No positive data found for country {country}, country specific scores set to 0.0")
            country_scores[country]["precision"].append(0)
            country_scores[country]["recall"].append(0)
            country_scores[country]["f1"].append(0)
        else:
            country_scores[country]["precision"].append(prec[1]) # Keep positive class only
            country_scores[country]["recall"].append(recall[1])
            country_scores[country]["f1"].append(f1[1])
    return country_scores


def eval_model(train_df, test_df, feat_index, clf, n_iter, sample_weights=None):
    X_train = train_df.iloc[:, feat_index:]
    y_train = train_df.LABEL
    X_test = test_df.iloc[:, feat_index:]
    y_test = test_df.LABEL

    country_results_temp = {c: {"f1": [], "precision": [], "recall": []} for c in test_df.COUNTRY.unique()}

    f1, prec, recall, importance = [], [], [], []
    for n in range(n_iter):
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = clf.predict(X_test)
        importance.append(clf.get_feature_importance())

        # Save results (positive class)
        prec_temp, recall_temp, f1_temp = precision_recall_fscore_support(y_test, y_pred)[:3]
        logging.debug(f"On iter {n} with f1: {f1_temp}")
        f1.append(f1_temp[1])
        prec.append(prec_temp[1])
        recall.append(recall_temp[1])

        # Country results
        get_country_scores(test_df, feat_index, clf, country_results_temp)

    country_results = {}
    for c, res in country_results_temp.items():
        country_results[c] = {
            "f1": {
                "avg": np.average(res["f1"]),
                "std": np.std(res["f1"])
            },
            "precision": {
                "avg": np.average(res["precision"]),
                "std": np.std(res["precision"])
            },
            "recall": {
                "avg": np.average(res["recall"]),
                "std": np.std(res["recall"])
            }
        }
    importance = np.array(importance)
    results = {
        "f1": {
            "avg": np.average(f1),
            "std": np.std(f1)
        },
        "precision": {
            "avg": np.average(prec),
            "std": np.std(prec)
        },
        "recall": {
            "avg": np.average(recall),
            "std": np.std(recall)
        },
        "feature_importance": {
            "avg": np.average(importance, axis=0) if importance.all() else None,
            "std": np.std(importance, axis=0) if importance.all() else None
        },
        "country_results": country_results
    }
    return results, clf  # Save last run of model for reference


def aggregate_data_weekly(daily_df, feature_index, num_features):
    """Aggregate the data by week"""
    logging.info("Aggregating data by week...")
    start = "2014-01-01"
    end = "2020-01-01"
    weeks = pd.date_range(start, end, freq="7D").values
    weekly_df = []
    
    for country, df in daily_df.groupby("COUNTRY"):
        start = time.time()
        df = df.set_index("date").sort_index()  # Set index to date to use .loc
        for i, week in enumerate(weeks[:-1]):
            temp = df.loc[week: weeks[i + 1]]
            row = [
                country,
                week,
                1 if 1 in temp.LABEL.values else 0  # True if there is at least one event during the week
            ]
            row.extend(temp.iloc[:, feature_index:].max().values)  # label, features
            weekly_df.append(row)
        end = time.time()
        logging.debug(f"Took {(end-start)/60:.2}s on country {country}")    
    weekly_df = pd.DataFrame(
        weekly_df,
        columns=["COUNTRY", "DATE", "LABEL"] + [f"feat_{i}" for i in range(num_features)]
    )
    return weekly_df
 

def augment_data_by_country(df):
    data_counts = dataset_df.COUNTRY.value_counts()
    max_val = data_counts.max()
    for country, count in data_counts.items():
        num_samples = max_val - data_counts[country]
        if num_samples > 0:
            sample_idx = np.random.choice(df[df['COUNTRY'] == country].index, num_samples)
            sample_idx = np.vectorize(lambda x: df.index.get_loc(x))(sample_idx)
            df = pd.concat([df, df.iloc[sample_idx]])
    return df


def load_dataset(args):
    """Load the dataset and format it according to commandline options"""
    logging.info("Loading, aggregating, and labelling features")
    dataset_filename = f"{args.output_dir}/{'weekly' if args.agg_weekly else 'daily'}_{args.lead_time}_{args.positive_day_threshold}_df.pkl.save"
    if args.from_save and os.path.exists(dataset_filename):
        logging.info(f"Loading dataset from {dataset_filename}")
        with open(dataset_filename, "rb") as f:
            dataset_df = pickle.load(f)
        feat_index = dataset_df.columns.values.tolist().index("LABEL") + 1  # Features start after labels
        return dataset_df, feat_index
    
    # Ground truth labels from ACLED
    acled_df = pd.read_csv(
        args.acled_event_data,
        keep_default_na=False,  # Preserve "NA" country code
        parse_dates=[0],  # Event dates
    ).set_index(["UN_code", "event_date"])
    original_length = len(acled_df)
    # Limit to entries that have more than positive_day_threshold events
    acled_df = acled_df[acled_df["count"].map(lambda x: x >= args.positive_day_threshold)]
    new_length = len(acled_df)
    logging.debug(f"Reduced from {original_length:,} events to {new_length:,} with positive day threshold {args.positive_day_threshold}")

    # Load features
    dataset_df = pd.read_csv(
        args.features,
        header=0 if args.keep_feature_names else None,
        index_col=[0, 1],  # country, date multi-index
        delimiter="\t",
        parse_dates=[1]
    )

    logging.info(dataset_df.head())
    if args.violent_demonstration_only:
        acled_df = acled_df[acled_df['sub_event_type'] == 'Violent demonstration']

    # Convert any string features to numbers with LabelEncoder
    # This fixes the "ValueError" for strings that sklearn classifiers throw
    encoder = LabelEncoder()
    for col, dtype in zip(dataset_df.columns, dataset_df.dtypes):
        if dtype == "object":
            dataset_df[col] = encoder.fit_transform(dataset_df[col])
    # Label the data according to ACLED ground truth
    dataset_df.insert(0, "LABEL", 0)
    dataset_df["LABEL"] = dataset_df.index.map(lambda x: 1 if x in acled_df.index else 0)
    dataset_df.index.names = ["COUNTRY", "DATE"]
    dataset_df.reset_index(inplace=True)
    feature_index = 3
    num_features = len(dataset_df.columns) - feature_index
    logging.debug(f"num features: {num_features}")

    if args.single_country is not None:
        dataset_df = dataset_df[dataset_df['COUNTRY'].map(lambda x: x in args.single_country)]
    if args.exclude_country is not None:
        dataset_df = dataset_df[dataset_df['COUNTRY'].map(lambda x: x not in args.exclude_country)]

    if args.agg_weekly:
        dataset_df = aggregate_data_weekly(dataset_df, feature_index, num_features)

    if args.lead_time != 0:
        logging.warning("Lead time not implemented yet")
        sys.exit(0)
        # Re-label dataset for forecasting
        # Want to predict 1 week into the future
        # Label the week before an event as positive
        # Remove the actual positive weeks
        # Save current labels
        labels = ground_truth 
        labels.reverse()
        logging.debug(labels)
        new_labels = [0 for i in range(len(labels))]
        labels_iter = iter(enumerate(labels))
        for i, label in labels_iter:
            # Negative case stays negative
            if label == 0:
                continue
            # Positive case. Remove the current positive sample and label the previous sample as positive
            new_labels[i] = -1
            new_labels[i+1] = 1
            # Skip the next iteration since we already labelled it
            next(labels_iter)
        logging.debug(new_labels)
        # Reverse the new labels list
        new_labels.reverse()
        dataset_df["LABEL"] = new_labels
        dataset_df = dataset_df[dataset_df.LABEL != -1]
        dataset_df.reset_index(drop=True, inplace=True)

    if args.save_df:
        logging.info(f"Saving labelled dataset to {dataset_filename}")
        with open(dataset_filename, "wb") as fo:
            pickle.dump(dataset_df, fo, protocol=4)
    return dataset_df, feature_index


def get_sample_weights(train_df):
    L = len(train_df)
    country_list = train_df.COUNTRY
    country_counts = dict(country_list.value_counts())
    weight_dict = {k:float(L)/v for k,v in country_counts.items()}
    # Let N denote number of countries, L denote number of training samples.
    # Suppose there is c_i samples for country i. So sum_i^N c_i = L
    # Using the above weight measure, w_i = L/c_i, so the total weight for
    # a single country i is c_i*w_i = L. Thus the total weight of the training
    # dataset is N*L.
    sample_weights = [weight_dict[x] for x in list(country_list)]
    return np.array(sample_weights)


def run_experiment(args, train_df, valid_df):
    # Run experiments
    for model_name in args.models:
        base_file_name = os.path.join(
            args.output_dir,
            f"results_{'weekly' if args.agg_weekly else 'daily'}_{args.lead_time}_{args.positive_day_threshold}_{model_name}")
        results_file = f"{base_file_name}_dict.pkl"
        model_file = f"{base_file_name}_model.pkl"
        if args.use_existing_model and os.path.exists(model_file):
            logging.info(f"Loading model from {model_file}")
            with open(model_file, "rb") as f:
                clf = pickle.load(f)
        else:
            clf = load_model(model_name, scale=args.fit_scaler, n_jobs=args.n_jobs)

        sample_weights = None
        if args.weight_by_country:
            sample_weights = get_sample_weights(train_df)

        if args.run_cv_train:
            results, clf = run_cv(
                train_df,
                feat_index,
                clf,
                args.random_seed,
                valid_df=valid_df,
                n_iter=args.n_iter,
                n_jobs=args.n_jobs,
                sample_weights=sample_weights)
        else:
            # USING VALIDATION SET INSTEAD OF TEST SET
            results, clf = eval_model(train_df, valid_df, feat_index, clf, args.n_iter, sample_weights)

        # Save model
        if not args.use_existing_model:
            pickle.dump(clf, open(model_file, "wb"), protocol=3)

        # Save results with the experiment settings
        # Add more information
        results["settings"] = vars(args)
        results["settings"]["current_model"] = model_name
        results["settings"]["num_train_examples"] = len(train_df)
        results["settings"]["num_test_examples"] = len(test_df)
        results["settings"]["num_validation_examples"] = len(valid_df)
        results["settings"]["num_positive_train_examples"] = train_df.LABEL.sum()
        results["settings"]["num_positive_test_examples"] = test_df.LABEL.sum()
        results["settings"]["num_positive_validation_examples"] = valid_df.LABEL.sum()
        results["settings"]["model_file"] = model_file
        with open(results_file, "wb") as f:
            pickle.dump(results, f, protocol=3)

        if args.run_cv_train:
            logging.info(f"""Summary results from model {model_name} with lead time {args.lead_time} with features {args.features}:
                \t\tF1\tPrecision\tRecall
                \tTrain CV:\t{results['f1']['avg']:.2} ({results['f1']['std']:.2})\t{results['precision']['avg']:.2} ({results['precision']['std']:.2})\t{results['recall']['avg']:.2} ({results['recall']['std']:.2})
                \tValidation:\t{results['validation_score']['f1']:.2}\t{results['validation_score']['precision']:.2}\t{results['validation_score']['recall']:.2}
                """)
        else:
            logging.info(f"""Summary results from model {model_name} with lead time {args.lead_time} with features {args.features}:
                \t\tF1\tPrecision\tRecall
                \tTrain:\t{results['f1']['avg']:.2} ({results['f1']['std']:.2})\t{results['precision']['avg']:.2} ({results['precision']['std']:.2})\t{results['recall']['avg']:.2} ({results['recall']['std']:.2})
                """)


if __name__ == "__main__":
    # Load commandline arguments
    args = parse_args()
    if args.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    # Load dataset and separate into train and test
    dataset_df, feat_index = load_dataset(args)
    train_df = dataset_df[dataset_df.DATE.map(lambda x: x.strftime("%Y") in args.train_years)]
    if args.balance_country_samples:
        train_df = augment_data_by_country(train_df)
    if args.time_variance_test:
        # get multiple test df
        # run eval_model on each train_df, test_df pair
        LATEST_YEAR = 2019
        last_train = int(args.train_years[-1])
        for year in range(last_train+1, LATEST_YEAR+1):
            year_df = dataset_df[dataset_df.DATE.map(lambda x: x.strftime("%Y") == str(year))]
            first_half = year_df[year_df.DATE.map(lambda x: int(x.strftime("%m")) <= 6)]
            second_half = year_df[year_df.DATE.map(lambda x: int(x.strftime("%m")) > 6)]
            print(f"Evaluating on first half of year {year}")
            run_experiment(args, train_df, first_half)
            print(f"Evaluating on second half of year {year}")
            run_experiment(args, train_df, second_half)
        exit(0)
    
    if args.validation_years:
        valid_df = dataset_df[dataset_df.DATE.map(lambda x: x.strftime("%Y") in args.validation_years)]
    else:
        valid_df = None
    test_df = dataset_df[dataset_df.DATE.map(lambda x: x.strftime("%Y") in args.test_years)]
    
    # Calculate rate of positive class
    train_pos_rate = train_df.LABEL.sum() / len(train_df)
    valid_pos_rate = valid_df.LABEL.sum() / len(valid_df)
    test_pos_rate = test_df.LABEL.sum() / len(test_df)
    logging.info(f"Training positive rate: {train_pos_rate:.3%} ({len(train_df):,}) "
                 f"and validation positive rate: {valid_pos_rate:.3%} ({len(valid_df):,})"
                 f"and testing positive rate: {test_pos_rate:.3%} ({len(test_df):,})")
    
    if args.test_on_training:
        run_experiment(args, train_df, train_df)
    elif args.test:
        run_experiment(args, train_df, test_df)
    else:
        run_experiment(args, train_df, valid_df)

