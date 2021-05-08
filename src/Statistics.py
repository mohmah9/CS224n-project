import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

df_text = pd.read_excel("../data/CLEAN/Clean_UTF8.xlsx")
df_words = pd.read_excel("../data/WORDS/Words_UTF8.xlsx")
df_sentences = pd.read_excel("../data/SENTENCES/Sentences_UTF8.xlsx")

print("Number of documents (All classes) : ",len(list(df_text["text"])))
classes = list(df_sentences['class'])
sentences= []
sentences_k=[]
sentences_h=[]
for i in range(len(list(df_sentences['sentences']))):
    t = list(df_sentences['sentences'])[i].split("', '")
    for j in t:
        sentences.append(j)
        if classes[i]=='keyfari':
            sentences_k.append(j)
        elif classes[i]=='hoghoghi':
            sentences_h.append(j)
print("Number of Sentences : ",len(sentences))
words=[]
words_k=[]
words_h=[]
for i in range(len(list(df_words['words']))):
    t = list(df_words['words'])[i].split("', '")
    for j in t:
        j = j.replace("['", "")
        words.append(j)
        if classes[i]=='keyfari':
            words_k.append(j)
        elif classes[i]=='hoghoghi':
            words_h.append(j)
print("Number of Words : ",len(words))

print("Number of unique words (tokens) : ",len(list(set(words))))

unique_sentences_k = set(sentences_k)
unique_sentences_h = set(sentences_h)
unique_words_h = set(words_h)
unique_words_k = set(words_k)

common_unique_words = unique_words_k & unique_words_h
print("Number of common unique words (tokens) : ",len(common_unique_words))
uncommon_unique_words = unique_words_k ^ unique_words_h
print("Number of uncommon unique words (tokens) : ",len(uncommon_unique_words))

uncommon_words_k = [x for x in words_k if x not in common_unique_words]
count_uncommon_words_k = Counter(uncommon_words_k)
uncommon_words_h = [x for x in words_h if x not in common_unique_words]
count_uncommon_words_h=Counter(uncommon_words_h)
print("uncommon words (tokens) keyfari: ",count_uncommon_words_k.most_common(10))
print("uncommon words (tokens) hoghoghi: ",count_uncommon_words_h.most_common(10))

common_words_k = [x for x in words_k if x in common_unique_words]
count_common_words_k = Counter(common_words_k)
common_words_h = [x for x in words_h if x in common_unique_words]
count_common_words_h=Counter(common_words_h)

h_relative_normalized_frequency = {}
k_relative_normalized_frequency = {}

for word in common_unique_words:
    h_relative_normalized_frequency[word]=(count_common_words_h[word]/len(words_h))/(count_common_words_k[word]/len(words_k))
    k_relative_normalized_frequency[word] = (count_common_words_k[word] / len(words_k)) / (count_common_words_h[word] / len(words_h))

h_relative_normalized_frequency=h_relative_normalized_frequency.items()
k_relative_normalized_frequency=k_relative_normalized_frequency.items()

sorted_h_relative_normalized_frequency= sorted(h_relative_normalized_frequency, key=lambda x: x[1],reverse=True)
sorted_k_relative_normalized_frequency= sorted(k_relative_normalized_frequency, key=lambda x: x[1],reverse=True)

print("Relative normalized frequency keyfari: ",sorted_k_relative_normalized_frequency[:10])
print("Relative normalized frequency hoghoghi: ",sorted_h_relative_normalized_frequency[:10])

def identity_tokenizer(text):
    return text

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, stop_words=["بر","این","آن","است","را",".","که","در","با","به","از"])
tfidf.fit_transform([words_k,words_h])
response1 = tfidf.transform([words_h])
feature_array = np.array(tfidf.get_feature_names())
tfidf_sorting = np.argsort(response1.toarray()).flatten()[::-1]
n = 10
top_n = feature_array[tfidf_sorting][:n]
print("TF-IDF for hoghoghi : ",top_n.tolist())
response2 = tfidf.transform([words_k])
feature_array = np.array(tfidf.get_feature_names())
tfidf_sorting = np.argsort(response2.toarray()).flatten()[::-1]
n = 10
top_n = feature_array[tfidf_sorting][:n]
print("TF-IDF for keyfari : ",top_n.tolist())


count_words = Counter(words)
count_words = count_words.most_common()
only_counts = [x[1] for x in count_words]
# print(only_counts[:10])

plt.figure(figsize=(25,8))
plt.title("Bars based on each word")
plt.xlabel("Words")
plt.ylabel("Ocurrence")
plt.bar(np.arange(len(only_counts[1::50])),only_counts[1::50],color="purple")
plt.show()
# plt.savefig("../data/statistics/histogram.png")