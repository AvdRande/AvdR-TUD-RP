from functools import partial
from pprint import pprint

import click
import numpy as np
import pandas as pd
from regex import E
import sklearn
from scipy import stats
import json
import random
import pickle

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

@click.command()
@click.option('--trainf', default='data\\tagrecomdata_topics220_repos152k_onehot_train.csv', prompt='train CSV file path', help='train CSV file path.')
@click.option('--testf', default='data\\tagrecomdata_topics220_repos152k_onehot_test.csv', prompt='test CSV file path', help='test CSV file path.')
@click.option('--hierarchyf', default='recommender\\clusters.json', prompt='tag hierarchy json file path', help='tag hierarchy json file path')
@click.option('--labels_column', default='labels', prompt='Topics Column name', help='The name of topics column.')
@click.option('--readme_column', default='text', prompt='Text Column name', help='The name of readme text column.')
@click.option('--model_output', default='HMC-LMLP', help='Model save path.')
@click.option('--learning_rate', default=0.05, help='Learning rate Value.')
@click.option('--epoch', default=100, help='Number of Epoch.')
@click.option('--word_ngrams', default=2, help='Number of wordNgrams.')
def ft(trainf, testf, hierarchyf, labels_column, readme_column, model_output, learning_rate, epoch, word_ngrams):
    train = pd.read_csv(trainf)
    test = pd.read_csv(testf)

    train_limiter = 4000
    test_limiter = 1000

    print("Now converting training csv to features and labels")
    train_features, train_labels = df2feature_class(train, train_limiter, readme_column, labels_column)
    
    print("Now converting testing csv to features and labels")
    test_features, test_labels = df2feature_class(test, test_limiter, readme_column, labels_column)

    def features_to_vectors(features_list):
        vectorizer = TfidfVectorizer(min_df=5, max_features=2500)
        all_features = [" ".join(f) for f in (features_list[0] + features_list[1])]
        vectors = vectorizer.fit_transform(all_features).toarray()
        return vectors[:len(features_list[0])], vectors[len(features_list[0]):]

    train_feature_vector, test_feature_vector = features_to_vectors([train_features, test_features])

    label_names = np.array(train.columns[:-2])

    if model_output == "LR":
        # classify using logistic regression
        clf = MultiOutputClassifier(LogisticRegression()).fit(train_feature_vector, train_labels)
        # result = clf.predict(train_feature_vector[:2])
        test_predictions = clf.predict(test_feature_vector)

        show_prec_rec_atn(test_predictions, test_labels)
        
    elif model_output == "HMC-LMLP":
        with open(hierarchyf) as f:


            print("Start training HMC-LMLP")
            hierarchy = json.load(f)

            # i_hier = build_idx_hierarchy(hierarchy, label_names)

            # print("Building subtrees for all of the labels")
            # label_subtrees = [labels_to_subtree(i_hier, tl) for tl in train_labels]

            depth = tree_depth(hierarchy)
            nn_layers = []
            predictions = []

            do_train = False
            filename = 'hmc-lmlp-model.sav'
            try:
                with open(filename, 'rb') as model_dump:
                    if input("Do you want to load the previous model?(y)") == "y":
                        nn_layers = pickle.load(model_dump)
                    else:
                        do_train = True
            except EnvironmentError:
                do_train = True

            if do_train:
                for level in range(depth):
                    print("Training HMC-LMLP layer ", level)
                    # the input matrix for this neural network
                    feature_matrix = train_feature_vector if level == 0 else predictions[level - 1]
                    # the expected outcome for this neural network
                    # target_labels = [labels_at_level(subtree, level) for subtree in label_subtrees] <- old method
                    target_labels = [target_labels_at_level(binlabels_to_text(label, label_names), hierarchy, level) for label in train_labels]
                    # how many hidden layers?
                    n_hidden_layers = max(min(100, len(feature_matrix[0])), len(target_labels[0]))
                    # make a new classifier for this level of the hierarchy
                    nn_layers.append(MLPClassifier(solver='lbfgs', hidden_layer_sizes=n_hidden_layers, random_state=1, max_iter=1000))
                    # train the neural network
                    print("Neural network consists of three layers: ", len(feature_matrix[0]), " to ", n_hidden_layers, " to ", len(target_labels[0]))
                    nn_layers[level].fit(feature_matrix, target_labels)
                    # add results to predictions
                    print("Fitting done, calculating predictions for next layer")
                    if level < depth - 1:
                        predictions.append(nn_layers[level].predict_proba(feature_matrix))

            if(input("Do you want to save the model?(y)") == "y"): 
                pickle.dump(nn_layers, open(filename, 'wb'))

            print("Training done, now showing test results: ")
            # test_predictions = nn_layers[3].predict(nn_layers[2].predict(nn_layers[1].predict(nn_layers[0].predict(test_feature_vector))))
            intermediate_predictions = []
            intermediate_predictions.append(nn_layers[0].predict_proba(test_feature_vector))
            for i in range(1, depth):
                intermediate_predictions.append(nn_layers[i].predict_proba(intermediate_predictions[-1]))
            test_predictions = intermediate_predictions[-1]

            # something to check, or postprocess: ensure that the hierarchical constrains are in fact obeyed

            show_prec_rec_atn(test_predictions, [target_labels_at_level(binlabels_to_text(test_label, label_names), hierarchy, depth-1) for test_label in test_labels], 1)

            show_prec_rec_atn(test_predictions, [target_labels_at_level(binlabels_to_text(test_label, label_names), hierarchy, depth-1) for test_label in test_labels], 3)

            show_prec_rec_atn(test_predictions, [target_labels_at_level(binlabels_to_text(test_label, label_names), hierarchy, depth-1) for test_label in test_labels], 5)


def show_prec_rec_atn(predictions, true_values, n):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(predictions)):
        top_n_label_idxs = np.argpartition(predictions[i], -n)[-n:]
        for idx in top_n_label_idxs:
            if true_values[idx] == 1:
                true_positives += 1
            else:
                false_positives += 1
    print("Precision@", n, ": ", true_positives / (true_positives + false_positives))

def target_labels_at_level(str_labels, hierarchy, level):
    if level == 0:
        # go into children and find in which children the labels are
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
    ft()