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

    limiter = 1000

    train_features, train_labels = df2feature_class(train, limiter, readme_column, labels_column)

    # convert features to vectors using tfidf
    vectorizer = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
    X = vectorizer.fit_transform([" ".join(train_feature) for train_feature in train_features]).toarray()

    if model_output == "LR":
        # classify using logistic regression
        clf = MultiOutputClassifier(LogisticRegression()).fit(X, train_labels)
        result = clf.predict(X[:2])
    elif model_output == "HMC-LMLP":
        with open(hierarchyf) as f:
            hierarchy = json.load(f)
            i_hier = build_idx_hierarchy(hierarchy, np.array(train.columns[:-2]))
            y = [labels_to_subtree(i_hier, tl) for tl in train_labels]
            
            print("Training first HMC-LMLP layer")
            # train the first layer of the hierarchy
            first_NN_layer = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(len(train.columns)-2), random_state=1)
            # first_y = np.array([np.array([1 if 1 in sub["content"] else 0 for sub in y2["children"]]) for y2 in y])
            first_y = [labels_at_level(yiter, 0) for yiter in y]
            first_NN_layer.fit(X, first_y)

            print("Training second HMC-LMLP layer")
            # train the second layer
            second_NN_layer = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(len(train.columns)-2), random_state=1)
            second_X = first_NN_layer.predict(X)
            second_y = [labels_at_level(yiter, 1) for yiter in y]
            second_NN_layer.fit(second_X, second_y)

            print(second_NN_layer.predict(first_NN_layer.predict(X[:100])))


    return 0

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

def labels_at_level(idx_hierarchy, level):
    if level == 0:
        return np.array([1 if 1 in sub["content"] else 0 for sub in idx_hierarchy["children"]])
    if level > 0:
        ret = []
        for child in idx_hierarchy["children"]:
            temp = labels_at_level(child, level - 1)
            for t in temp:
                ret.append(t)
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