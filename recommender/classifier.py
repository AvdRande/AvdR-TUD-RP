import warnings
import click
import numpy as np
import pandas as pd
from scipy import stats
import json
import pickle
from os.path import exists

from sklearn.metrics import average_precision_score, roc_curve, auc
import keras

from helper_fun import *

import hmc_lmlp
import lr
import awx
import hmc_lmlp_imp
import chmcnnh
import hmcnf


@click.command()
@click.option('--trainf', default='data\\tagrecomdata_topics220_repos152k_onehot_train.csv', prompt='train CSV file path', help='train CSV file path.')
@click.option('--testf', default='data\\tagrecomdata_topics220_repos152k_onehot_test.csv', prompt='test CSV file path', help='test CSV file path.')
@click.option('--hierarchyf', default='recommender\\hierarchies\\AC_COM_30-5v2.json', prompt='tag hierarchy json file path', help='tag hierarchy json file path')
@click.option('--save-or-load', prompt='Choose whether to save the model (s), load the previous model (l), or neither', help='The name of topics column.')
@click.option('--model_type', default='HMCN-F', prompt='Model save path. Options: LR, HMC-LMLP, HMC-LML-imp, HMCN-F, AWX, C-HMCNN(h)')
@click.option('--labels_column', default='labels', help='The name of topics column.')
@click.option('--readme_column', default='text', help='The name of readme text column.')
@click.option('--learning_rate', default=0.05, help='Learning rate Value.')
@click.option('--epochs', default=128, help='Number of Epoch.')
@click.option('--word_ngrams', default=2, help='Number of wordNgrams.')
def classify(trainf, testf, hierarchyf, save_or_load, labels_column, readme_column, model_type, learning_rate, epochs, word_ngrams):
    train = pd.read_csv(trainf)
    test = pd.read_csv(testf)

    hierarchy = json.load(open(hierarchyf))
    depth = tree_depth(hierarchy)

    train_limiter = 2500
    test_limiter = 500

    if save_or_load != "l":
        print("Now converting training csv to features and labels")
    train_features, train_labels = df2feature_class(
        train, train_limiter, readme_column, labels_column)

    print("Now converting testing csv to features and labels")
    test_features, test_labels = df2feature_class(
        test, test_limiter, readme_column, labels_column)

    train_feature_vector, test_feature_vector = features_to_vectors(
        [train_features, test_features], n_features=2500)

    label_names = np.array(train.columns[:-2])

    model = []

    do_train = False
    filename = 'recommender/models/' + model_type + '-model.sav'
    if save_or_load == "l":
        print("Loading previous model", model_type)
        if model_type == "AWX":
            model = keras.models.load_model(filename[:-4])
        else:
            if exists(filename):
                model_dump = open(filename, 'rb')
                model = pickle.load(model_dump)
    else:
        do_train = True

    rec = None
    if model_type == "LR":
        rec = lr
    if model_type == "HMC-LMLP":
        rec = hmc_lmlp
    if model_type == "HMC-LMLP-imp":
        rec = hmc_lmlp_imp
    if model_type == "HMCN-F":
        rec = hmcnf
    if model_type == "AWX":
        rec = awx
    if model_type == "C-HMCNN(h)":
        rec = chmcnnh

    if do_train:
        print("Start training ", model_type)
        model = rec.train(train_feature_vector, train_labels,
                          hierarchy, label_names, epochs)

    if save_or_load == "s":
        # rec.save(filename)
        if model_type == "AWX":
            model.save(filename[:-4])
        else:
            pickle.dump(model, open(filename, 'wb'))

    print("Training done, now predicting")

    test_predictions = []

    test_predictions = rec.predict(model, test_feature_vector, depth)

    target_labels = test_labels

    # test_predictions = np.interp(test_predictions, (test_predictions.min(), test_predictions.max()), (0, 1))

    from PIL import Image
    im_data = np.zeros((len(test_predictions), len(test_predictions[0])))
    for i in range(len(im_data)):
        for j in range(len(im_data[0])):
            im_data[i][j] = test_predictions[i][j] * 255
    im = Image.fromarray(im_data).convert("L")
    im.save("filename.jpeg")

    if model_type == "LR":
        target_labels = (test_labels.T[model[1]]).T
    if model_type in {"HMC-LMLP", "HMC-LMLP-imp", "HMCN-F", "AWX", "C-HMCNN(h)"}:
        target_labels = map_labels_to_tree_order(
            test_labels, hierarchy, label_names)

    print("AUPCR:", average_precision_score(target_labels, test_predictions))

    p, r, f = prf_at_k(target_labels, test_predictions, [1, 3, 5])

    print(p)
    print(r)
    print(f)

    loop_through_predictions(target_labels, test_predictions, label_names)

    # for i in range(1, 6, 2):
    #     show_prec_rec_atn(test_predictions, target_labels, i, th_from_youden(map_labels_to_tree_order(train_labels, hierarchy, label_names)[:train_prediction_limit], train_predictions))


def loop_through_predictions(orig, pred, names):
    for i in range(len(orig)):
        orig_names = [names[j] for j in range(len(orig[i])) if orig[i][j] == 1]
        n_origs = len(orig_names)
        ind = np.argpartition(pred[i], -n_origs)[-n_origs:]
        pred_names = names[ind]
        print("Original:", orig_names, " vs Predictions:", pred_names)
        if i % 5 == 0:
            input("Press ENTER for more predictions")

# extracted from https://www.kaggle.com/code/willstone98/youden-s-j-statistic-for-threshold-determination/notebook


def th_from_youden(labels, predictions):
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()

    n_labels = len(labels[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_labels):
            label_slice = np.array(labels)[:, i]
            pred_slice = np.array(predictions)[:, i]
            fpr[i], tpr[i], thresholds[i] = roc_curve(
                label_slice, pred_slice, drop_intermediate=False)
            roc_auc[i] = auc(fpr[i], tpr[i])

    J_stats = [None] * n_labels
    opt_thresholds = [None] * n_labels

    # Compute Youden's J Statistic for each class
    for i in range(n_labels):
        J_stats[i] = tpr[i] - fpr[i]
        opt_thresholds[i] = thresholds[i][np.argmax(J_stats[i])]

    return np.average(opt_thresholds)

# adapted from https://surprise.readthedocs.io/en/latest/FAQ.html#how-to-compute-precision-k-and-recall-k


def show_prec_rec_atn(predictions, true_values, n, threshold):
    pred_and_true = {
        i: list(zip(predictions[i], true_values[i])) for i in range(len(predictions))}

    precisions = dict()
    recalls = dict()

    for i, tags in pred_and_true.items():
        tags.sort(key=lambda x: x[0], reverse=True)

        recommended = [(pred >= threshold) for (pred, _) in tags[:n]]
        relevant = [(true_v == 1) for (_, true_v) in tags]

        n_rel_and_rec = [((true_v == 1) and (pred >= threshold))
                         for (pred, true_v) in tags[:n]]

        precisions[i] = sum(n_rel_and_rec) / \
            sum(recommended) if sum(recommended) != 0 else 0
        recalls[i] = sum(n_rel_and_rec) / \
            sum(relevant) if sum(relevant) != 0 else 0

    print("Precision@", n, ": ",
          sum(prec for prec in precisions.values()) / len(precisions))
    print("Recall@", n, ": ", sum(rec for rec in recalls.values()) / len(recalls))


def df2feature_class(dataframe, n, feature_column, label_column):
    total = n
    running = 0

    df_features = []
    df_labels = []
    for _, row in dataframe.iterrows():
        # print(len(row[readme_column].split(" ")))
        df_features.append(row[feature_column].split(" "))

        tlabels_list = row[label_column].strip("[]").split(" ")
        dupe = []
        for x in tlabels_list:
            if len(x) > 0:
                if x[0].isnumeric():
                    dupe.append(int(x[0]))
        df_labels.append(np.array(dupe))
        running += 1
        if running > total - 1:
            break
    return df_features, np.array(df_labels)


if __name__ == "__main__":
    classify()
