import csv
import numpy as np
import json
import random
sim_matrix = np.zeros((220, 220))

headers = []

with open("data/tagrecomdata_topics220_repos152k_onehot_train.csv", "r") as tagdatafile: #, newline=''
    td_reader = csv.reader(tagdatafile, delimiter=",", quotechar="\"")


    total = 121389
    running = 0

    # print(next(td_reader, None))
    headers = next(td_reader, None)

    for row in td_reader:
        running += 1
        if random.uniform(0, 1) < 100 / total:
            print(round(running / total,2))
        labels = row[:-2]
        for label_cur in range(len(labels)):
            for label_scan in range(len(labels)):
                if label_cur != label_scan and labels[label_cur] == "1" and labels[label_scan] == "1": # the labels are different but both appear in the matrix
                    sim_matrix[label_scan, label_cur] += 1

        if running > total-1:
            break

with open("data/tagrecomdata_topics220_repos152k_onehot_test.csv", "r") as tagdatafile: #, newline=''
    td_reader = csv.reader(tagdatafile, delimiter=",", quotechar="\"")


    total = 30348
    running = 0

    # print(next(td_reader, None))
    # headers = next(td_reader, None)

    for row in td_reader:
        running += 1
        if random.uniform(0, 1) < 100 / total:
            print(round(running / total,2))
        labels = row[:-2]
        for label_cur in range(len(labels)):
            for label_scan in range(len(labels)):
                if label_cur != label_scan and labels[label_cur] == "1" and labels[label_scan] == "1": # the labels are different but both appear in the matrix
                    sim_matrix[label_scan, label_cur] += 1

        if running > total-1:
            break

sim_as_dict = {}
sim_as_dict["headers"] = headers
for i in range(220):
    sim_as_dict[headers[i]] = sim_matrix[i].tolist()

json_file = open("tree_generation/json_simatrix.json", "w")
json.dump(sim_as_dict, json_file)
