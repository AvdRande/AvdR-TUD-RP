import wikipediaapi
import string

wiki_wiki = wikipediaapi.Wikipedia('en')

page_py = wiki_wiki.page('Python_(programming_language)')
page_java = wiki_wiki.page('Java_(programming_language)')

# use TF-IDF in later stage
print(page_py.summary)
def tf(summary):
    clean_summary = summary.lower().split()
    dic = {}
    for word in clean_summary:
        if word in dic:
            dic[word] = dic[word] + 1
        else:
            dic[word] = 1
    return dic

print(tf(page_py.summary))

def cos(a, b):
    a_sorted = dict(sorted(a.items(), key=lambda item: item[1], reverse=True))
    b_sorted = dict(sorted(b.items(), key=lambda item: item[1], reverse=True))
    n = min(len(a), len(b))
    return []

cos(tf(page_py.summary), tf(page_java.summary))