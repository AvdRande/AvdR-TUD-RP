import json
# from recommender.helper_fun import *

ag_simatrix_tree = json.load(open("recommender\\hierarchies\\four_hier\\SKG-BK.json"))

def get_clusters_at_level(tree, level):
    ret = []
    if level == 0:
        for child in tree["children"]:
            ret.append(child["content"])
    else:
        for child in tree["children"]:
            for clusters in get_clusters_at_level(child, level - 1):
                ret.append(clusters)
    return ret

clusters = get_clusters_at_level(ag_simatrix_tree, 0)
print(len(clusters))
# input()
for cluster in clusters:
    print(len(cluster))