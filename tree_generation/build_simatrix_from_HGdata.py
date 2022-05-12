import csv
import numpy as np
import json
import random

with open("data/tagrecomdata_topics220_repos152k_onehot_train.csv", "r") as tagdatafile: #, newline=''
    td_reader = csv.reader(tagdatafile, delimiter=",", quotechar="\"")

    sim_matrix = np.zeros((220, 220))

    total = 121389
    running = 0

    # print(next(td_reader, None))
    headers = next(td_reader, None)

    for row in td_reader:
        running += 1
        if random.uniform(0, 1) < 0.01:
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

    # printstring = ""
    # for i in range(len(sim_matrix)):
    #     for j in range(len(sim_matrix[0])):
    #         printstring += str(sim_matrix[i, j]) + " "
    #     printstring += "\n"
    # f = open("tree_generation/simatrix.txt", "w")
    # f.write(printstring)
    # f.close()
