import warnings
import click
import numpy as np
import pandas as pd
from responses import target
from scipy import stats
import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc

import hmc_lmlp
import lr

@click.command()
@click.option('--trainf', default='data\\tagrecomdata_topics220_repos152k_onehot_train.csv', prompt='train CSV file path', help='train CSV file path.')
@click.option('--testf', default='data\\tagrecomdata_topics220_repos152k_onehot_test.csv', prompt='test CSV file path', help='test CSV file path.')
@click.option('--hierarchyf', default='recommender\\hierarchies\\best_cluster.json', prompt='tag hierarchy json file path', help='tag hierarchy json file path')
@click.option('--save-or-load', prompt='Choose whether to save the model (s), load the previous model (l), or neither', help='The name of topics column.')
@click.option('--labels_column', default='labels', help='The name of topics column.')
@click.option('--readme_column', default='text', help='The name of readme text column.')
@click.option('--model_output', default='LR', help='Model save path.')
@click.option('--learning_rate', default=0.05, help='Learning rate Value.')
@click.option('--epoch', default=100, help='Number of Epoch.')
@click.option('--word_ngrams', default=2, help='Number of wordNgrams.')
def classify(trainf, testf, hierarchyf, save_or_load, labels_column, readme_column, model_output, learning_rate, epoch, word_ngrams):
    train = pd.read_csv(trainf)
    test = pd.read_csv(testf)

    hierarchy = json.load(open(hierarchyf))
    depth = tree_depth(hierarchy)

    train_limiter = 1000
    test_limiter = 200

    if save_or_load != "l":
        print("Now converting training csv to features and labels")
    train_features, train_labels = df2feature_class(train, train_limiter, readme_column, labels_column)
    
    print("Now converting testing csv to features and labels")
    test_features, test_labels = df2feature_class(test, test_limiter, readme_column, labels_column)

    def features_to_vectors(features_list):
        vectorizer = TfidfVectorizer(
            max_features=2500,
            stop_words='english',
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',
            ngram_range=(1, 2)
        )
        all_features = [" ".join(f) for f in (features_list[0] + features_list[1])]
        vectors = vectorizer.fit_transform(all_features).toarray()
        return vectors[:len(features_list[0])], vectors[len(features_list[0]):]

    train_feature_vector, test_feature_vector = features_to_vectors([train_features, test_features])

    label_names = np.array(train.columns[:-2])

    model = []

    do_train = False
    filename = 'recommender/models/' + model_output + '-model.sav'
    try:
        with open(filename, 'rb') as model_dump:
            if save_or_load == "l":
                print("Loading previous model ", model_output)
                model = pickle.load(model_dump)
            else:
                do_train = True
    except EnvironmentError:
        do_train = True

    if do_train:
        print("Start training ", model_output)
        if model_output == "LR":
            model = lr.train(train_feature_vector, train_labels)
        if model_output == "HMC-LMLP":
            model = hmc_lmlp.train(train_feature_vector, train_labels, hierarchy, label_names)
        
    if save_or_load == "s": 
        pickle.dump(model, open(filename, 'wb'))

    print("Training done, now predicting")

    test_predictions = []
    train_predictions = [] # used for Youden J threshold finding
    train_prediction_limit = 100

    if model_output == "LR":
        test_predictions = lr.predict(model[0], test_feature_vector)
        train_predictions = lr.predict(model[0], train_feature_vector[:train_prediction_limit])
    if model_output == "HMC-LMLP":
        test_predictions = hmc_lmlp.predict(model, test_feature_vector, depth)
        train_predictions = hmc_lmlp.predict(model, train_feature_vector[:train_prediction_limit], depth)

    target_labels = test_labels

    if model_output == "LR":
        target_labels = (test_labels.T[model[1]]).T
    if model_output == "HMC-LMLP":
        target_labels = [target_labels_at_level(binlabels_to_text(test_label, label_names), hierarchy, depth-1) for test_label in test_labels]

    for i in range(1, 6, 2):
        show_prec_rec_atn(test_predictions, target_labels, i, th_from_youden(train_labels[:train_prediction_limit], train_predictions))
        
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
            fpr[i], tpr[i], thresholds[i] = roc_curve(labels[:, i], predictions[:, i], drop_intermediate=False)
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
    pred_and_true = {i : list(zip(predictions[i], true_values[i])) for i in range(len(predictions))}

    precisions = dict()
    recalls = dict()

    for i, tags in pred_and_true.items():
        tags.sort(key=lambda x: x[0], reverse=True)

        recommended = [(pred >= threshold) for (pred, _) in tags[:n]]
        relevant = [(true_v == 1) for (_, true_v) in tags]

        n_rel_and_rec = [((true_v == 1) and (pred >= threshold)) for (pred, true_v) in tags[:n]]

        precisions[i] = sum(n_rel_and_rec) / sum(recommended) if sum(recommended) != 0 else 0
        recalls[i] = sum(n_rel_and_rec) / sum(relevant) if sum(relevant) != 0 else 0

    print("Precision@", n, ": ", sum(prec for prec in precisions.values()) / len(precisions))
    print("Recall@", n, ": ", sum(rec for rec in recalls.values()) / len(recalls))

def target_labels_at_level(str_labels, hierarchy, level):
    if level == 0:
        # go through list of children and find for which children the labels are present
        ret = []
        if len(hierarchy["children"]) == 0:
            for tag in hierarchy["content"]:
                ret.append(1 if tag in str_labels else 0)
        else:
            for child in hierarchy["children"]:
                any_label_found = False
                for label in str_labels:
                    if is_label_in_tree(label, child):
                        any_label_found = True
                ret.append(1 if any_label_found else 0)
        return ret
    else:
        ret = []
        for child in hierarchy["children"]:
            ret += target_labels_at_level(str_labels, child, level - 1)
        return ret

def is_label_in_tree(label, hierarchy):
    return label in hierarchy["content"] or any([is_label_in_tree(label, child) for child in hierarchy["children"]])

# convert hierarchy to hierarchy of indexes of the labels
def build_idx_hierarchy(hierarchy, label_names):
    ret = {}

    ret["name"] = hierarchy["value"]["id"]

    ret["content"] = []
    for content in hierarchy["content"]:
        ret["content"].append(np.where(label_names == content)[0][0])
    
    ret["children"] = []
    for child in hierarchy["children"]:
        ret["children"].append(build_idx_hierarchy(child, label_names))

    return ret

# find the subtree in the idx hierarchy for a given labellist of a repo
# it will have empty branches for all the parts of the tree where it doesn't "go"
def labels_to_subtree(idx_hierarchy, labels):
    ret = {}
    
    ret["name"] = idx_hierarchy["name"]
    
    if len(idx_hierarchy["children"]) == 0:
        ret["content"] = [labels[idx] for idx in idx_hierarchy["content"]]
        ret["children"] = []
    else:
        ret["content"] = []
        for i in range(len(labels)):
            if labels[i] == 1 and i in idx_hierarchy["content"]:
                ret["content"].append(1)
            else:
                ret["content"].append(0)
        
        ret["children"] = []
        for child in idx_hierarchy["children"]:
            ret["children"].append(labels_to_subtree(child, labels))

    return ret

def tree_depth(tree):
    if len(tree["children"]) == 0:
        return 1
    else:
        return 1 + tree_depth(tree["children"][0])

def labels_at_level(idx_hierarchy, level):
    if level == 0:
        if len(idx_hierarchy["children"]) == 0:
            return idx_hierarchy["content"]
        else:
            return np.array([1 if 1 in sub["content"] else 0 for sub in idx_hierarchy["children"]])
    if level > 0:
        ret = []
        for child in idx_hierarchy["children"]:
            temp = labels_at_level(child, level - 1)
            for t in temp:
                ret.append(t)
        return ret

def get_leaves(hierarchy):
    ret = []
    if len(hierarchy["children"]) == 0:
        return hierarchy["content"]
    else:
        for child in hierarchy["children"]:
            ret += get_leaves(child)
    return ret

def binlabels_to_text(binlabels, label_names):
    ret = []
    for i in range(len(binlabels)):
        if binlabels[i] == 1:
            ret.append(label_names[i])
    return ret

def df2feature_class(dataframe, n, feature_column, label_column):
    total = n
    running = 0

    df_features = []
    df_labels = []
    for _, row in dataframe.iterrows():
        # print(len(row[readme_column].split(" ")))
        df_features.append(row[feature_column].split(" ")) #take first 20 features to ensure homogeniosity
        
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