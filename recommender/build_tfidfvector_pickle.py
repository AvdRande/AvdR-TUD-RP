import pickle
import pandas as pd
from classifier import df2feature_class
from helper_fun import features_to_vectors


def main():
    trainf = 'data/tagrecomdata_topics220_repos152k_onehot_train.csv'
    testf = 'data/tagrecomdata_topics220_repos152k_onehot_test.csv'

    train_limiter = 20000
    test_limiter = 4000

    n_features = 10000

    readme_column = 'text'
    labels_column = 'labels'

    train = pd.read_csv(trainf)
    test = pd.read_csv(testf)

    print("Converting csv to feature lists and labels")
    train_features, train_labels = df2feature_class(
        train, train_limiter, readme_column, labels_column)
    test_features, test_labels = df2feature_class(
        test, test_limiter, readme_column, labels_column)

    print("Converting features to TF-IDF vectors")
    train_feature_vector, test_feature_vector = features_to_vectors(
        [train_features, test_features], n_features=n_features)

    pickle.dump((train_feature_vector, train_labels), open("data/train_feature_pickle.sav", 'wb'))
    pickle.dump((test_feature_vector, test_labels), open("data/test_feature_pickle.sav", 'wb'))

if __name__ == '__main__':
    main()
