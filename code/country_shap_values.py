'''
Run SHAP algorithms to get feature importance value on country prediction model

Author: Jack Zhang
'''

import argparse
import pickle
import os
import logging

import shap
import numpy as np

from predict_country import load_dataset, dataset_split

def parse_args():
    parser = argparse.ArgumentParser()
    # Data and labels
    parser.add_argument("--model-path", type=str,  required=True, help="Path to classifier to evaluate")
    parser.add_argument("--args-path", type=str,  required=True, help="Path to arguments that paired with the classifier")
    parser.add_argument("--output-dir", help="Folder to save results", required=True)
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to estimate feature SHAP weight")
    parser.add_argument("--countries-from-list", nargs="?", type=str, default=None)
    parser.add_argument("--countries", nargs="+", type=str, default=None, help="Run on a list of countries")
    parser.add_argument("--output-name", type=str, default=None, help="Specify output filename")
    parser.add_argument("--top-output", type=int, default=None, help="Number of top SHAP features to show")
    parser.add_argument("--shap-limit", type=float, default=0.0)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.countries_from_list is not None:
        countries = []
        with open(args.countries_from_list, "r") as f:
            for line in f:
                countries.append(line.strip())
        args.countries = countries
    return args

def get_shap_values_tree(clf, X_test, feature_names, num_samples=500, top_output=None, shap_lim=0.0):
    explainer = shap.TreeExplainer(clf)
    X_sample = X_test.sample(min(num_samples, len(X_test)))
    shap_values = explainer.shap_values(X_sample)
    values = np.array(shap_values) # (num_classes, num_samples, num_features)
    feature_weight = list(np.abs(values).sum(axis=0).mean(axis=0))
    if top_output is not None:
        weight_list = [(y,x) for x,y in sorted(zip(feature_weight, feature_names), reverse=True) if x > shap_lim][:top_output]
    else:
        weight_list = [(y,x) for x,y in sorted(zip(feature_weight, feature_names), reverse=True) if x > shap_lim]
    return weight_list

if __name__ == "__main__":
    args = parse_args()
    print(args.countries)
    with open(args.model_path, "rb") as f:
        clf = pickle.load(f)
    with open(args.args_path, "rb") as f:
        clf_args = pickle.load(f)
    
    out_name = args.output_name if args.output_name is not None else "shap_weight.out"

    dataset_df, label_index, feature_index, encoder = load_dataset(clf_args)
    train_df, valid_df, test_df = dataset_split(dataset_df, clf_args)

    if args.countries is not None:
        with open(os.path.join(args.output_dir, out_name), "w") as f:
            for country in args.countries:
                f.write(f"{country}\n")
                country_idx = encoder.transform([country])[0]
                if valid_df is not None:
                    country_df = valid_df.loc[valid_df.COUNTRY == country_idx]
                else:
                    country_df = train_df.loc[train_df.COUNTRY == country_idx]

                if country_df.empty:
                    logging.info(f"Skipping country {country} due to insufficient data")
                    continue

                weight_list = get_shap_values_tree(
                    clf.clf[0], country_df.iloc[:, feature_index:], country_df.columns[feature_index:], 
                    args.num_samples, args.top_output, args.shap_limit)
                for name, weight in weight_list:
                    f.write(f"{name}\t{weight:.4f}\n")
        
        exit(0)

    if valid_df is not None:
        weight_list = get_shap_values_tree(
            clf.clf[0], valid_df.iloc[:, feature_index:], valid_df.columns[feature_index:], 
            args.num_samples, args.top_output, args.shap_limit)
    else:
        weight_list = get_shap_values_tree(
            clf.clf[0], valid_df.iloc[:, feature_index:], valid_df.columns[feature_index:], 
            args.num_samples, args.top_output, args.shap_limit)

    with open(os.path.join(args.output_dir, out_name), "w") as f:
        for name, weight in weight_list:
            f.write(f"{name}\t{weight:.4f}\n")