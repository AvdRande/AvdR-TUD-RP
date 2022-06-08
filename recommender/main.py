import json

import numpy as np
from helper_fun import add_parents_to_labels, make_anc_matrix


def main():
    print("test")
    hierarchyf = "recommender\\hierarchies\\AC_COM_30-5v2.json"
    hierarchy = json.load(open(hierarchyf))
    
    train_labels_with_parents = np.array([add_parents_to_labels(
        train_label, hierarchy, label_names) for train_label in train_labels])
    h_m = make_anc_matrix(hierarchy, len(train_labels_with_parents[0]))


if __name__ == "__main__":
    main()
