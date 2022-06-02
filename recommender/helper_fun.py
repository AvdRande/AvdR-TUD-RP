from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy import stats

        
# from izadi code
def prf_at_k(y_original, y_pred_probab, k_list):
    r, p, f = {}, {}, {}
    y_org_array = np.array(y_original)

    for k in k_list:
        org_label_count = np.sum(y_org_array, axis=1).tolist()
        
        topk_ind = np.argpartition(y_pred_probab, -1 * k, axis=1)[:, -1 * k:]
        pred_in_orig = y_org_array[np.arange(y_org_array.shape[0])[:, None], topk_ind]
        common_topk = np.sum(pred_in_orig, axis=1)
        recall, precision, f1 = [], [], []
        for index, value in enumerate(common_topk):
            recall.append(value / min(k, org_label_count[index]))
            precision.append(value / k)
        r.update({'R@' + str(k): "{:.2f}".format(np.mean(recall) * 100)})
        p.update({'P@' + str(k): "{:.2f}".format(np.mean(precision) * 100)})
        f1 = stats.hmean([precision, recall])
        f.update({'F@' + str(k): "{:.2f}".format(np.mean(f1) * 100)})
    return r, p, f


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

def map_labels_to_tree_order(labels, hierarchy, label_names):
    return [target_labels_at_level(binlabels_to_text(test_label, label_names), hierarchy, tree_depth(hierarchy) - 1) for test_label in labels]

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

def features_to_vectors(features_list, n_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=n_features,
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