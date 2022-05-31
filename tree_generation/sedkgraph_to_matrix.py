import json
from matplotlib.font_manager import json_load
import pandas as pd
import numpy as np

file_name = "tree_generation\\sedkgraph-updated.xlsx"

df_skg = pd.read_excel(io=file_name, sheet_name="sedkgraph_relationships among t", engine='openpyxl').to_numpy()

skg_vertices = [row[0] for row in pd.read_excel(io=file_name, sheet_name="all_topics", engine='openpyxl').to_numpy()]

skg = {}
index = 0
for relation in df_skg:
    if relation[0] not in skg:
        skg[relation[0]] = []
    if relation[2] not in skg:
        skg[relation[2]] = []
    skg[relation[0]].append(relation[2])
    
    skg[relation[2]].append(relation[0])

required_tags = json_load("tree_generation\\json_simatrix.json")["headers"][:-2]

n_tags = len(required_tags)

matrix = {
    "headers": required_tags
}

def dijkstra(skg, source, vertices, target_vertices):
    dist = {}
    prev = {}
    Q = set()

    for vertex in vertices:
        dist[vertex] = 999999
        prev[vertex] = 999999
        Q.add(vertex)

    dist[source] = 0

    while len(Q) > 0:
        u = ""
        d= 9999999
        for vertex in Q:
            if dist[vertex] < d:
                u = vertex
                d = dist[vertex]
        Q.remove(u)

        for v in Q:
            if v in skg[u]:
                alt = dist[u] + 1
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
    # return [dist[vertices[i]] for i in range(len(vertices))]
    return [dist[target_vertex] for target_vertex in target_vertices]


for i in range(n_tags): #rows
    matrix[required_tags[i]] = dijkstra(skg, required_tags[i], skg_vertices, required_tags)
    print("Distances found for", required_tags[i])

# find max value
maxdist = -1
i = 0
for row in matrix:
    if i == 0:
        i = 1
    else:
        for val in matrix[row]:
            if val > maxdist and val < 1000:
                maxdist = val

# limit all values to maxdist + 1

i = 0
for row in matrix:
    if i == 0:
        i = 1
    else:
        for j in range(len(matrix[row])):
            if matrix[row][j] > maxdist:
                matrix[row][j] = maxdist

print(matrix)
json_file = open("tree_generation/sedkgraph_distance.json", "w+")
json.dump(matrix, json_file)
