#!/usr/bin/env python3

from datetime import datetime
import numpy as np
import pandas as pd
import json
from multiprocessing.pool import Pool

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

    with open("partial_results.txt", "a") as result_file:
        for result_line in result:
            result_file.write(result_line + "\n")
        result_file.write("\n")


def train_and_predict_model(tp_params):
    trained_model = train_model(tp_params[0])
    predictions = predict_model(trained_model, tp_params[1])
    model_name = tp_params[2][2]
    hier_name = tp_params[0][3]["name"]
    save_predict_results(predictions, tp_params[2], model_name, hier_name)
    return predictions


def main():
    trainf = 'data/tagrecomdata_topics220_repos152k_onehot_train.csv'
    testf = 'data/tagrecomdata_topics220_repos152k_onehot_test.csv'

    hierarchy_paths = ["recommender/hierarchies/four_hier/" + hier_name +
                       ".json" for hier_name in ["COM-AC", "COM-BK", "SKG-AC", "SKG-BK"]]

    readme_column = 'text'
    labels_column = 'labels'
    epochs = 256

    print("Reading CSVs")

    train = pd.read_csv(trainf)
    test = pd.read_csv(testf)

    hierarchies = [json.load(open(hierarchyf))
                   for hierarchyf in hierarchy_paths]

    train_limiter = len(train)
    test_limiter = len(test)

    n_features = 20000

    # train_limiter = 100
    # test_limiter = 20

    # n_features = 50

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
        # lr,
        # hmc_lmlp,
        # hmc_lmlp_imp,
        hmcnf,
        # awx,
        # chmcnnh
    ]

    train_and_pred_params = []

    for hierarchy in hierarchies:
        hierarchy_mapping = make_hierarchy_mapping(hierarchy)
        for rec in recommenders:
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
                tree_depth(hierarchy)
            ),
                (
                test_labels,
                hierarchy_mapping,
                rec.get_name()
            )))

    results = []

    for hier_path in hierarchy_paths:
        for rec in recommenders:
            results.append([
                "Hier: " + hier_path.split("/")[-1][:-5] +
                " Rec: " + rec.get_name()])

    # clear partial result file
    with open("partial_results.txt", "w") as result_file:
        result_file.write("")

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
