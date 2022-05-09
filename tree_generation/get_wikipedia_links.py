import pandas as pd
import numpy as np


from googlesearch import search

def search_for(query):
    for result in search(query, tld="com", num=1, stop=1):
        return result

file_name = "sedkgraph.xlsx"
pd_all_topics = pd.read_excel(io=file_name, sheet_name="all_topics", engine='openpyxl')

print(pd_all_topics.loc[1, "topic"])
for i in range(863):
    if str(pd_all_topics.loc[i, "Link"]) == "nan":
        result = search_for("wikipedia" + pd_all_topics.loc[i, "topic"])
        print("Found: ", result, " for ", pd_all_topics.loc[i, "topic"])
        pd_all_topics.loc[i, "Link"] = result

        pd_all_topics.to_excel(file_name, sheet_name="all_topics")
