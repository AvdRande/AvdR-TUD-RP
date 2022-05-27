import keras

from AWX.awx_core.layers import *
from tensorflow import keras

# mostly copied from AWX/awx_example.py
def train(train_feature_vector, train_labels, hierarchy, label_names):
    print("Building AWX model")
    train_labels_with_parents = np.array([add_parents_to_labels(train_label, hierarchy, label_names) for train_label in train_labels]) # this is very slow atm, no surprise
    hierarchy_matrix = make_hier_matrix(hierarchy, len(train_labels_with_parents[0]))
    nfeatures = len(train_feature_vector[0])
    
    model = keras.models.Sequential([
        keras.layers.Dense(
            nfeatures,
            activation='tanh',
            kernel_regularizer=keras.regularizers.l1_l2(l2=0, l1=0),
            name='dense_1'
        ),
        keras.layers.GaussianNoise(0.1),
        keras.layers.Dense(
            nfeatures,
            activation='tanh',
            kernel_regularizer=keras.regularizers.l1_l2(l2=0, l1=0),
            name='dense_2'
        ),
        keras.layers.GaussianNoise(0.1),
        AWX(
            A=hierarchy_matrix, 
            n_norm=1, 
            activation='sigmoid', 
            kernel_regularizer=keras.regularizers.l1_l2(l1=0, l2=1e-6), 
            name='AWX'
        )
    ], "AWXClassifier")
    model.compile(
        keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['binary_crossentropy']
    )
    print("Training AWX model")
    model.fit(
        train_feature_vector,
        train_labels_with_parents,
        epochs=256,
        batch_size=32,
        initial_epoch=0,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, monitor='loss', mode='auto', ),
        ]
    )
    return model

def predict(model, test_feature_vector):
    predict = model.predict(test_feature_vector)
    return [row[-220:] for row in predict]

def add_parents_to_labels(labels, hierarchy, label_names):
    names = [label_names[i] for i in range(len(labels)) if labels[i] == 1]
    ret = [1]
    next_level = hierarchy["children"]
    leaf_labels = []
    while len(next_level) > 0:
        upcoming_level = []
        for child in next_level:
            if any(name in child["content"] for name in names):
                ret.append(1)
            else:
                ret.append(0)
            if len(child["children"]) == 0:
                leaf_labels += [1 if label in names else 0 for label in child["content"]]
            else:
                upcoming_level += child["children"]
        next_level = upcoming_level
    ret += leaf_labels
    return ret

def make_hier_matrix(hierarchy, ret_size):
    ret = np.zeros((ret_size, ret_size))
    node_names = find_node_names(hierarchy)

    next_level = [hierarchy]
    while len(next_level) > 0:
        upcoming_level = []
        for node in next_level:
            node_idx = node_names.index(node["value"]["uniqueId"])
            if len(node["children"]) == 0:
                for leaf in node["content"]:
                    leaf_idx = node_names.index(leaf)
                    ret[leaf_idx][node_idx] = 1
            else:
                for child in node["children"]:
                    child_idx = node_names.index(child["value"]["uniqueId"])
                    ret[child_idx][node_idx] = 1
                upcoming_level += node["children"]
        next_level = upcoming_level

    from PIL import Image
    im = Image.fromarray(ret * 255).convert("L")
    im.save("filename.jpeg")
    return ret

def find_node_names(hierarchy):
    ret = [hierarchy["value"]["uniqueId"]]
    next_level = hierarchy["children"]
    leaf_labels = []
    while len(next_level) > 0:
        upcoming_level = []
        for child in next_level:
            ret.append(child["value"]["uniqueId"])
            if len(child["children"]) == 0:
                leaf_labels += child["content"]
            else:
                upcoming_level += child["children"]
        next_level = upcoming_level
    ret += leaf_labels
    return ret