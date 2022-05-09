from queue import Queue
import pandas as pd
import numpy as np

# read excel sheet, get list of relation types and relations
# save both of those as dicts

file_name = "sedkgraph.xlsx"
sheet = ["sedkgraph_relationships among t", "relation_types", "github_featured topics"]

df_graph_array = pd.read_excel(io=file_name, sheet_name=sheet[0]).to_numpy()
df_relations_array = pd.read_excel(io=file_name, sheet_name=sheet[1]).to_numpy()
df_topics_array = pd.read_excel(io=file_name, sheet_name=sheet[2]).to_numpy()

df_graph = [{}] * len(df_graph_array)
index = 0
for relation in df_graph_array:
    df_graph[index] = {
        "LHS": str(relation[0]),
        "relation_type": str(relation[1]),
        "RHS": str(relation[2])
    }
    index += 1

df_relations = {}
for relation_type in df_relations_array:
    df_relations[relation_type[0]] = [int(i) for i in relation_type[1:]]

df_topics = []
for relation in df_graph:
    df_topics.append(str(relation["LHS"]))
    df_topics.append(str(relation["RHS"]))

df_topics = list(dict.fromkeys(df_topics))

# define distance metric between tags using distance in relation graph
# damn who thought I'd actually implement A* at this time (22:39)
# haha this^ idiot thought he needed pathfinding, but it's just breadth-first search. dumbass

def tag_distance(tag_a, tag_b, df_graph, df_topics, df_relations):
    if tag_a < 0 or tag_b < 0:
        return 0 #?

    Q = Queue() # the queue contains the tags
    distance = {k: 9999999 for k in df_topics} # this contains a dictionary for each tag and their distance to tag_a
    visited_tags = set() # this is a set of tags that have been visited
    Q.put(df_topics[tag_a])
    visited_tags.update({df_topics[tag_a]})
    
    while not Q.empty():
        vertex = Q.get()
        print(Q.qsize())
        if vertex == df_topics[tag_a]:
            distance[vertex] = 0
        for relation in df_graph:
            found_relation = False
            lhs = ""
            rhs = ""
            if relation["LHS"] == vertex and relation["RHS"] not in visited_tags:
                lhs = relation["LHS"]
                rhs = relation["RHS"]
                found_relation = True
            if relation["RHS"] == vertex and relation["LHS"] not in visited_tags:
                lhs = relation["RHS"]
                rhs = relation["LHS"]
                found_relation = True

            relation_weight = [df_relations[r][1] for r in df_relations if r == relation["relation_type"]][0]
            if found_relation:
                if distance[rhs] > distance[vertex] + relation_weight:
                    distance[rhs] = distance[vertex] + relation_weight
            
                if rhs == df_topics[tag_b]:
                    return distance[rhs]

                Q.put(rhs)
                visited_tags.update(lhs)
                found_relation = False
            
print(df_topics)

print(tag_distance(0, 100, df_graph, df_topics, df_relations))

# run algorithm from http://ilpubs.stanford.edu:8090/775/1/2006-10.pdf

# notes: the algorithm requires sorting of tags, based on their centrality? whatever, that won't be a problem later
# I guess it could be sorted by the amount of times the tag appears in the relation graph?
# def build_tree(df_Graph, df_topics, df_relations):
#     tree = set({(-1, 0)})
#     for i in range(len(df_topics)):
#         topic_i = df_topics[i]
#         max_candidate_val = 0
#         for all 

# print result?