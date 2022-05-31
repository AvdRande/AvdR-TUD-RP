from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def train(train_feature_vector, train_labels):
    used_labels = []
    for i, col in enumerate(train_labels.T):
        if sum(col) > 0:
            used_labels.append(i)

    clf = MultiOutputClassifier(LogisticRegression(n_jobs=-1, class_weight='balanced', verbose=2)).fit(train_feature_vector, (train_labels.T[used_labels]).T)

    return [clf, np.array(used_labels)]

def predict(clf, test_feature_vector):
    # return clf.predict(test_feature_vector)
    predict_classes = clf.predict_proba(test_feature_vector)

    ret = np.zeros((len(predict_classes[0]), len(predict_classes)))

    for y in range(len(ret)):
        for x in range(len(ret[0])):
            ret[y][x] = predict_classes[x][y][1]

    return ret