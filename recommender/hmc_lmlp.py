from helper_fun import tree_depth, target_labels_at_level, binlabels_to_text
from sklearn.neural_network import MLPClassifier

def train(train_feature_vector, train_labels, hierarchy, label_names, epochs):
    nn_layers = []
    predictions = []
    
    depth = tree_depth(hierarchy)

    for level in range(depth):
            print("Training HMC-LMLP layer ", level)
            # the input matrix for this neural network
            feature_matrix = train_feature_vector if level == 0 else predictions[level - 1]
            # the expected outcome for this neural network
            # target_labels = [labels_at_level(subtree, level) for subtree in label_subtrees] <- old method
            target_labels = [target_labels_at_level(binlabels_to_text(label, label_names), hierarchy, level) for label in train_labels]
            # how many hidden layers?
            n_hidden_layers = max(len(feature_matrix[0]), len(target_labels[0]))
            # make a new classifier for this level of the hierarchy
            nn_layers.append(MLPClassifier(solver='lbfgs', hidden_layer_sizes=n_hidden_layers, random_state=1, max_iter=epochs, verbose=False))
            # train the neural network
            print("Neural network consists of three layers: ", len(feature_matrix[0]), " to ", n_hidden_layers, " to ", len(target_labels[0]))
            nn_layers[level].fit(feature_matrix, target_labels)
            # add results to predictions
            print("Fitting done, calculating predictions for next layer")
            if level < depth - 1:
                    predictions.append(nn_layers[level].predict_proba(feature_matrix))
    return nn_layers

def predict(model, test_feature_vector, depth):
    intermediate_predictions = []
    intermediate_predictions.append(model[0].predict_proba(test_feature_vector))
    for i in range(1, depth):
        intermediate_predictions.append(model[i].predict_proba(intermediate_predictions[-1]))
    return intermediate_predictions[-1]

def get_name():
    return "HMC-LMLP"