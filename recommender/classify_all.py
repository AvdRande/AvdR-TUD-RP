#!/usr/bin/env python3

from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import json
from multiprocessing.pool import Pool
import sys
from os.path import exists

from sklearn.metrics import average_precision_score

from classifier import df2feature_class
from helper_fun import features_to_vectors, prf_at_k, tree_depth, make_hierarchy_mapping

import lr
import hmc_lmlp
import hmc_lmlp_imp
import awx
import chmcnnh
import hmcnf

import build_tfidfvector_pickle


def train_model(t_params):
    return t_params[0](t_params[1], t_params[2], t_params[3], t_params[4], t_params[5])


def predict_model(model, p_params):
    return p_params[0](model, p_params[1], p_params[2])


def save_predict_results(prediction, s_params, model_name, hier_name):
    test_labels = s_params[0]
    ordered_labels = []
    hier_map = s_params[1]
    if "LR" != model_name:
        for row in test_labels:
            ordered_labels.append([row[hier_map[i]] for i in range(len(row))])
    result = []
    result.append("Model: " + model_name + ", Hierarchy: " + hier_name)
    result.append("Time finished: " +
                  datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    result.append(
        "AUPCR: " + str(average_precision_score(ordered_labels, prediction)))

    k_values = [1, 2, 3, 4, 5, 7, 10]

    r, p, f = prf_at_k(ordered_labels, prediction, k_values)

    result.append(
        ", ".join(["P@" + str(k) + ": " + p["P@" + str(k)] for k in k_values]))
    result.append(
        ", ".join(["R@" + str(k) + ": " + r["R@" + str(k)] for k in k_values]))
    result.append(
        ", ".join(["F@" + str(k) + ": " + f["F@" + str(k)] for k in k_values]))

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


def main(hier_idx, rec_idx):
    if hier_idx > 0 and rec_idx == 0:
        return

    hier_names = ["COM-AC", "COM-BK", "SKG-AC", "SKG-BK"]
    hier_name = hier_names[hier_idx]
    hierarchy_path = "recommender/hierarchies/four_hier/" + hier_name + ".json"

    hierarchy = json.load(open(hierarchy_path))

    label_names = hierarchy["content"]
    hier_depth = tree_depth(hierarchy)

    recommender = [
        lr,
        hmc_lmlp,
        # hmc_lmlp_imp,
        hmcnf,
        awx,
        chmcnnh
    ][rec_idx]

    rec_name = recommender.get_name()

    epochs = 1

    if not exists("data/train_feature_pickle.sav"):
        print("Making tf idf vector pickles")
        build_tfidfvector_pickle.main()

    (train_feature_vector, train_labels) = pickle.load(
        open("data/train_feature_pickle.sav", "rb"))
    (test_feature_vector, test_labels) = pickle.load(
        open("data/test_feature_pickle.sav", "rb"))

    print("Loaded pickles")

    print("Training recommender", rec_name, "with hierarchy", hier_name)

    rec_model = recommender.train(
        train_feature_vector[:1000], train_labels[:1000], hierarchy, label_names, epochs)

    print("Done training recommender")

    print("Making predictions")

    prediction = recommender.predict(
        rec_model, test_feature_vector, hier_depth)

    print("Done with predictions")

    print("Re-ordering predicted labels")

    hier_mapping = make_hierarchy_mapping(hierarchy)
    re_ordered_preds = [[row[hier_mapping[i]] for i in range(len(row))] for row in prediction]

    results = []

    results.append(
        "AUPCR: " + str(average_precision_score(test_labels, re_ordered_preds)))

    k_values = [1, 3, 5]

    r, p, f = prf_at_k(test_labels, re_ordered_preds, k_values)

    results.append(
        ", ".join(["P@" + str(k) + ": " + p["P@" + str(k)]for k in k_values]))
    results.append(
        ", ".join(["R@" + str(k) + ": " + r["R@" + str(k)]for k in k_values]))
    results.append(
        ", ".join(["F@" + str(k) + ": " + f["F@" + str(k)]for k in k_values]))

    with open("results.txt", "a") as result_file:
        for result_line in results:
            result_file.write(result_line + "\n")
        result_file.write("\n")


if __name__ == '__main__':
    # hier_idx, rec_idx = int(sys.argv[1]), int(sys.argv[2])
    hier_idx, rec_idx = 0, 3
    main(hier_idx, rec_idx)
