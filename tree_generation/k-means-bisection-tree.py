import json
from sklearn.cluster import BisectingKMeans
import numpy as np


n_clusters = 20

matrix = json.load(open("tree_generation/co-occurence-distance.json"))

headers = matrix["headers"]

X = np.array([np.array(matrix[label_name]) for label_name in headers[:-2]])

cluster_idx = BisectingKMeans(
    n_clusters=n_clusters, 
    random_state=0,
    init='k-means++', 
    bisecting_strategy="largest_cluster",
    n_init=16
).fit_predict(X)

cluster = {
    'value': {
        'id': "root"
    },
    'content': headers,
    'children': [{
        'value': {'id': "lvl " + str(i)},
        'content': [],
        'children': []
    } for i in range(n_clusters)]
}

for i in range(len(headers[:-2])):
    cluster['children'][cluster_idx[i]]["content"].append(headers[i])

print(cluster)
json.dump(cluster, open("bisecting_cluster.json", "w"))