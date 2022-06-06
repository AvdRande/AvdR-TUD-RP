import json
import helper_fun
from helper_fun import get_lvlsizes_from_tree

def main():
    print("test")
    hierarchyf = "recommender\\hierarchies\\AC_COM_30-5v2.json"
    hierarchy = json.load(open(hierarchyf))

    print(get_lvlsizes_from_tree(hierarchy))

if __name__ == "__main__":
    main()