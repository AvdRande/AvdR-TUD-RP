import json
from sklearn.cluster import BisectingKMeans
import numpy as np


n_clusters = [130, 60, 20]

matrix = json.load(open("tree_generation/co-occurence-distance.json"))

headers = matrix["headers"]

X = np.array([np.array(matrix[label_name]) for label_name in headers[:-2]])

cluster_idxs = [[i] for i in range(len(headers[:-2]))]

for cluster_size in n_clusters:
    cluster_idx = BisectingKMeans(
        n_clusters=cluster_size, 
        random_state=0,
        init='k-means++', 
        bisecting_strategy="largest_cluster",
        n_init=16
    ).fit_predict(X)
    for i in range(len(cluster_idx)):
        cluster_idxs[i].append(cluster_idx[i])

# cluster = {
#     'value': {
#         'id': "root"
#     },
#     'content': headers,
#     'children': [None] * n_clusters[-1]
# }

# first make lowest level clusters
clusters = [None] * n_clusters[0]

for row in cluster_idxs:
    if clusters[row[1]] == None:
        clusters[row[1]] = {
            'value': {
                'id': int(row[1]),
                'uniqueId': "lvl2-"+str(row[1])},
            'content': [headers[row[0]]],
            'children': []
        } 
    else:
        clusters[row[1]]["content"].append(headers[row[0]])

next_clusters = [None] * n_clusters[1]

for cluster in clusters:
    c_idx = cluster['value']['id']
    parent_cluster_idx = [row[2] for row in cluster_idxs if row[1] == c_idx][0]

    if next_clusters[parent_cluster_idx] == None:
        next_clusters[parent_cluster_idx] = {
            'value': {
                'id': int(parent_cluster_idx),
                'uniqueId': "lvl1-"+str(parent_cluster_idx)},
            'content': cluster["content"].copy(),
            'children': [cluster]
        }
    else:
        next_clusters[parent_cluster_idx]["content"] += cluster["content"].copy()
        next_clusters[parent_cluster_idx]["children"].append(cluster)

# repeat party for highest tier hierarchy

next_next_clusters = [None] * n_clusters[2]

for cluster in next_clusters:
    c_idx = cluster['value']['id']
    parent_cluster_idx = [row[3] for row in cluster_idxs if row[2] == c_idx][0]

    if next_next_clusters[parent_cluster_idx] == None:
        next_next_clusters[parent_cluster_idx] = {
            'value': {
                'id': int(parent_cluster_idx),
                'uniqueId': "lvl0-"+str(parent_cluster_idx)},
            'content': cluster["content"].copy(),
            'children': [cluster]
        }
    else:
        next_next_clusters[parent_cluster_idx]["content"] += cluster["content"].copy()
        next_next_clusters[parent_cluster_idx]["children"].append(cluster)

final_cluster = {
    'value' : {
        'id': 0,
        'uniqueId': 'root'
    },
    'content': headers[:-2],
    'children': next_next_clusters
}

print('debug')

json.dump(final_cluster, open("bisecting_cluster.json", "w"))