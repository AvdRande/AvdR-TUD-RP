import json
 
# Opening JSON file
f = open('tree_generation/clusters.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
def remove_trivial(tree):
    for child_node in tree["children"]:
        remove_trivial(child_node)
        if len(child_node["children"]) == 1:
            return child_node["children"][0]
    return tree

data_no_trivial = remove_trivial(data)
print("debug")

with open('tree_generation/nt-clusters.json', 'w') as outfile:
    json.dump(data_no_trivial, outfile)

def only_leaves(tree):
    ret = True
    for child_node in tree["children"]:
        if len(child_node["children"]) > 0:
            ret = False
    return ret

def print_leaf_clusters(tree):
    if only_leaves(tree):
        printable = ""
        for child in tree["children"]:
            printable += child["content"][0] + ", "
        print("[", printable[:-2], "]")
    else:
        for child in tree["children"]:
            print_leaf_clusters(child)

# print_leaf_clusters(data_no_trivial) doesnt work

def list_parents(tree, running_heritage, heritage_list):
    if len(heritage_list) > 100:
        return heritage_list
    if len(tree["children"]) == 0:
        if len(heritage_list) == 0:
            heritage_list.append(running_heritage +  [tree["content"][0]])
        elif heritage_list[-1][-1] != [tree["content"][0]]:
            heritage_list.append(running_heritage +  [tree["content"][0]])
    else:
        for child in tree["children"]:
            heritage_list = heritage_list + list_parents(child, running_heritage + [child["content"][0]], heritage_list)
    return heritage_list

def list_parents_bfs(tree):
    heritage_lists = []
    queue = [(tree, ["root"])]
    
    while len(queue) > 0:
        cur = queue.pop(0)
        for child in cur[0]["children"]:
            if len(child["children"]) == 0: # if node
                heritage_lists.append(cur[1] + child["content"][0])



parents = list_parents(data_no_trivial, ["root"], [])

print(parents[0])
print(parents[-1])

print("debug")

# Closing file
f.close()