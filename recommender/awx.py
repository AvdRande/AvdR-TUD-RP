import keras

from AWX.awx_core.layers import *
from tensorflow import keras

from helper_fun import *

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
            nfeatures/2,
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
        epochs=420,
        batch_size=50,
        initial_epoch=0,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, monitor='loss', mode='auto', ),
        ]
    )
    return model

def predict(model, test_feature_vector):
    predictions = model.predict(test_feature_vector)
    return [row[-220:] for row in predictions] # the last 220 labels are actually the leaves