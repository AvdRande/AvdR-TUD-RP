
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