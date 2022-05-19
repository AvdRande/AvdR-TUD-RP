from functools import partial
from pprint import pprint

import click
import fasttext
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
import json
import random

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

    limiter = 10000

    train_features, train_labels = df2feature_class(train, limiter, readme_column, labels_column)

    # convert features to vectors using tfidf
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.7, max_features=2500)
    X = vectorizer.fit_transform([" ".join(train_feature) for train_feature in train_features]).toarray()

    label_names = np.array(train.columns[:-2])

    if model_output == "LR":
        # classify using logistic regression
        clf = MultiOutputClassifier(LogisticRegression()).fit(X, train_labels)
        result = clf.predict(X[:2])
    elif model_output == "HMC-LMLP":
        with open(hierarchyf) as f:
            hierarchy = json.load(f)

            i_hier = build_idx_hierarchy(hierarchy, label_names)

            y = [labels_to_subtree(i_hier, tl) for tl in train_labels]

            depth = tree_depth(hierarchy)
            nn_layers = []
            predictions = []
            for level in range(depth):
                print("Training HMC-LMLP layer ", level)
                # the input matrix for this neural network
                feature_matrix = X if level == 0 else predictions[level - 1]
                # the expected outcome for this neural network
                target_labels = [labels_at_level(yiter, level) for yiter in y]
                # how many hidden layers?
                n_hidden_layers = max(min(100,len(feature_matrix[0])), len(target_labels[0]))
                # make a new classifier for this level of the hierarchy
                nn_layers.append(MLPClassifier(solver='lbfgs', hidden_layer_sizes=n_hidden_layers, random_state=1, max_iter=100))
                # train the neural network
                print("Neural network consists of three layers: ", len(feature_matrix[0]), " to ", n_hidden_layers, " to ", len(target_labels[0]))
                nn_layers[level].fit(feature_matrix, target_labels)
                # add results to predictions
                print("Fitting done, calculating predictions for next layer")
                if level == depth - 1:
                    predictions.append(nn_layers[level].predict(feature_matrix))
                else:
                    predictions.append(nn_layers[level].predict_proba(feature_matrix))

            print("Training done, now showing result:")
            actual = [np.array(labels_at_level(labels_to_subtree(i_hier, train_label), depth - 1)) for train_label in train_labels[:100]]
            for i in range(100):
                print("Prediction: ", binlabels_to_text(predictions[depth - 1][i], label_names))
                print("Actual: " , binlabels_to_text(actual[i], label_names))
                print("Difference: ", sum(np.abs(predictions[depth - 1][i] - np.array(actual[i]))))
                input()

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
            return idx_hierarchy["content"] # CONTINUE HERE STUPID ME!
        else:
            return np.array([1 if 1 in sub["content"] else 0 for sub in idx_hierarchy["children"]])
    if level > 0:
        ret = []
        for child in idx_hierarchy["children"]:
            temp = labels_at_level(child, level - 1)
            for t in temp:
                ret.append(t)
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

        if random.uniform(0, 1) < 100/total:
            print(round(running / total,2))
        running += 1
        if running > total - 1:
            break
    return df_features, np.array(df_labels)

if __name__ == "__main__":
    ft()