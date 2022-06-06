import json
from helper_fun import make_hierarchy_mapping


def main():
    print("test")
    hierarchyf = "recommender\\hierarchies\\AC_COM_30-5v2.json"
    hierarchy = json.load(open(hierarchyf))

    print(make_hierarchy_mapping(hierarchy))


if __name__ == "__main__":
    main()
