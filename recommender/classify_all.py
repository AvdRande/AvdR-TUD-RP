#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
from multiprocessing.pool import Pool

import multiprocessing

from sklearn.metrics import average_precision_score

from classifier import df2feature_class
from helper_fun import features_to_vectors, prf_at_k, tree_depth

import lr
import hmc_lmlp
import hmc_lmlp_imp
import awx
import chmcnnh


def train_model(t_params):
    return t_params[0](t_params[1], t_params[2], t_params[3], t_params[4], t_params[5])

def predict_model(p_params):
    return p_params[1][0](p_params[0], p_params[1][1], p_params[1][2])

def train_and_predict_model(tp_params):
    print("Now training a model yey")
    trained_model = train_model(tp_params[0])
    predictions = predict_model((trained_model, tp_params[1]))
    return predictions

def main():
    trainf = 'data\\tagrecomdata_topics220_repos152k_onehot_train.csv'
    testf = 'data\\tagrecomdata_topics220_repos152k_onehot_test.csv'
    
    hierarchy_paths = ["recommender\\hierarchies\\four_hier\\" + hier_name + ".json" for hier_name in ["COM-AC", "COM-BK", "SKG-AC", "SKG-BK"]]
    
    readme_column = 'text'
    labels_column = 'labels'
    epochs = 1

    print("Reading CSVs")

    train = pd.read_csv(trainf)
    test = pd.read_csv(testf)

    hierarchies = [json.load(open(hierarchyf)) for hierarchyf in hierarchy_paths]
  
    train_limiter = len(train)
    test_limiter = len(test)

    n_features = 20000

    print("Converting csv to feature lists and labels")
    train_features, train_labels = df2feature_class(train, train_limiter, readme_column, labels_column)
    test_features, test_labels = df2feature_class(test, test_limiter, readme_column, labels_column)

    print("Converting features to TF-IDF vectors")
    train_feature_vector, test_feature_vector = features_to_vectors([train_features, test_features], n_features=n_features)

    label_names = np.array(train.columns[:-2])

    recommenders = [
        lr,
        hmc_lmlp, 
        hmc_lmlp_imp, 
        awx, 
        chmcnnh
    ]

    train_and_pred_params = []

    for hierarchy in hierarchies:
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
            )))

    results = []

    for hier_path in hierarchy_paths:
        for rec in recommenders:
            results.append([
                "Hier: " + hier_path.split("\\")[-1][:-5] +\
                    " Rec: " + rec.get_name()])

    pool = multiprocessing.Pool(processes = len(train_and_pred_params))
    predictions = pool.map(train_and_predict_model, train_and_pred_params)

    for i, prediction in enumerate(predictions):
        results[i].append("AUPCR: " + str(average_precision_score(test_labels, prediction)))

        k_values = [1, 2, 3, 4, 5, 7, 10]

        r, p, f = prf_at_k(test_labels, prediction, k_values)

        results[i].append(", ".join(["P@" + str(k) + ": " + p["P@" + str(k)]for k in k_values]))
        results[i].append(", ".join(["R@" + str(k) + ": " + r["R@" + str(k)]for k in k_values]))
        results[i].append(", ".join(["F@" + str(k) + ": " + f["F@" + str(k)]for k in k_values]))

    with open("results.txt", "w") as result_file:
        for result in results:
            for result_line in result:
                result_file.write(result_line + "\n")
            result_file.write("\n")
            

if __name__ == '__main__':
    main()