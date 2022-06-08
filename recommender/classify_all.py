#!/usr/bin/env python3

from datetime import datetime
import numpy as np
import pandas as pd
import json
from multiprocessing.pool import Pool
import sys

from sklearn.metrics import average_precision_score

from classifier import df2feature_class
from helper_fun import features_to_vectors, prf_at_k, tree_depth, make_hierarchy_mapping

import lr
import hmc_lmlp
import hmc_lmlp_imp
import awx
import chmcnnh
import hmcnf


def train_model(t_params):
    return t_params[0](t_params[1], t_params[2], t_params[3], t_params[4], t_params[5])


def predict_model(model, p_params):
    return p_params[0](model, p_params[1], p_params[2])


def save_predict_results(prediction, s_params, model_name, hier_name):
    test_labels = s_params[0]
    if "LR" not in s_params[2]:
        for row in test_labels:
            row = [row[s_params[1][i]] for i in range(len(row))]
    result = []
    result.append("Model: " + model_name + ", Hierarchy: " + hier_name)
    result.append("Time finished: " +
                  datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    result.append(
        "AUPCR: " + str(average_precision_score(test_labels, prediction)))

    k_values = [1, 2, 3, 4, 5, 7, 10]

    r, p, f = prf_at_k(test_labels, prediction, k_values)

    result.append(
        ", ".join(["P@" + str(k) + ": " + p["P@" + str(k)]for k in k_values]))
    result.append(
        ", ".join(["R@" + str(k) + ": " + r["R@" + str(k)]for k in k_values]))
    result.append(
        ", ".join(["F@" + str(k) + ": " + f["F@" + str(k)]for k in k_values]))

    with open(s_params[3], "a") as result_file:
        for result_line in result:
            result_file.write(result_line + "\n")
        result_file.write("\n")


def train_and_predict_model(tp_params):
    model_name = tp_params[2][2]
    hier_name = tp_params[0][3]["name"]
    print("Start training", model_name, "with", hier_name,
          "at", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    trained_model = train_model(tp_params[0])
    print("Start predicting", model_name, "with", hier_name,
          "at", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    predictions = predict_model(trained_model, tp_params[1])
    print("Saving results for", model_name, "with", hier_name,
          "at", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    save_predict_results(predictions, tp_params[2], model_name, hier_name)
    return predictions


def main():
    trainf = 'data/tagrecomdata_topics220_repos152k_onehot_train.csv'
    testf = 'data/tagrecomdata_topics220_repos152k_onehot_test.csv'

    hierarchy_paths = ["recommender/hierarchies/four_hier/" + hier_name +
                       ".json" for hier_name in ["COM-AC", "COM-BK", "SKG-AC", "SKG-BK"]]

    readme_column = 'text'
    labels_column = 'labels'

    print("Reading CSVs")

    train = pd.read_csv(trainf)
    test = pd.read_csv(testf)

    hierarchies = [json.load(open(hierarchyf))
                   for hierarchyf in hierarchy_paths]

    train_limiter = 10000
    test_limiter = 2000

    n_features = 5000

    epochs = 128

    if len(sys.argv) > 1:
        train_limiter = len(train)
        test_limiter = len(test)

        n_features = 25000

        epochs = 256

    print("Converting csv to feature lists and labels")
    train_features, train_labels = df2feature_class(
        train, train_limiter, readme_column, labels_column)
    test_features, test_labels = df2feature_class(
        test, test_limiter, readme_column, labels_column)

    print("Converting features to TF-IDF vectors")
    train_feature_vector, test_feature_vector = features_to_vectors(
        [train_features, test_features], n_features=n_features)

    label_names = np.array(train.columns[:-2])

    recommenders = [
        lr,
        hmc_lmlp,
        # hmc_lmlp_imp,
        hmcnf,
        awx,
        chmcnnh
    ]

    partial_file_name = "partial_results.txt" if len(
        sys.argv) == 1 else "partial_results_full.txt"

    # clear partial result file
    with open(partial_file_name, "w") as result_file:
        result_file.write("")

    train_and_pred_params = []

    for hier_i, hierarchy in enumerate(hierarchies):
        hierarchy_mapping = make_hierarchy_mapping(hierarchy)
        t_depth = tree_depth(hierarchy)
        for rec in recommenders:
            # make sure to only launch one LR, as more would be unneccesary
            if not (hier_i > 0 and rec.get_name() == "LR"):
                train_and_pred_params.append(((
                    # first the training data
                    rec.train,
                    train_feature_vector,
                    train_labels,
                    hierarchy,
                    label_names,
                    epochs
                ),
                    (
                    rec.predict,
                    test_feature_vector,
                    t_depth
                ),
                    (
                    test_labels,
                    hierarchy,
                    rec.get_name(),
                    partial_file_name
                )))
    results = []

    for hier_path in hierarchy_paths:
        for rec in recommenders:
            results.append([
                "Hier: " + hier_path.split("/")[-1][:-5] +
                " Rec: " + rec.get_name()])

    pool = Pool(processes=len(train_and_pred_params))
    predictions = pool.map(train_and_predict_model, train_and_pred_params)

    for i, prediction in enumerate(predictions):
        results[i].append(
            "AUPCR: " + str(average_precision_score(test_labels, prediction)))

        k_values = [1, 2, 3, 4, 5, 7, 10]

        r, p, f = prf_at_k(test_labels, prediction, k_values)

        results[i].append(
            ", ".join(["P@" + str(k) + ": " + p["P@" + str(k)]for k in k_values]))
        results[i].append(
            ", ".join(["R@" + str(k) + ": " + r["R@" + str(k)]for k in k_values]))
        results[i].append(
            ", ".join(["F@" + str(k) + ": " + f["F@" + str(k)]for k in k_values]))

    with open("results.txt", "w") as result_file:
        for result in results:
            for result_line in result:
                result_file.write(result_line + "\n")
            result_file.write("\n")


if __name__ == '__main__':
    main()
