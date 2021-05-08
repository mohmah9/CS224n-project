import json
import csv
import time
import pandas as pd
from hazm import *

print("Pre Processing ...")
start = time.time()
with open('keyfari.json') as f:
    data_k = json.load(f)
with open('hoghoghi.json') as f:
    data_h = json.load(f)

r_k=[]
for i in data_k:
    if len(i['raay']) < 150 and len(i['raay2']) < 150:
        r_k.append(i)
for i in r_k:
    data_k.remove(i)
for i in range(len(data_k)):
    if len(data_k[i]['raay2']) < 150 and len(data_k[i]['raay']) > 150:
        data_k[i]['final_raay']= data_k[i]['raay']
    elif len(data_k[i]['raay']) < 150 and len(data_k[i]['raay2']) > 150:
        data_k[i]['final_raay'] = data_k[i]['raay2']
    elif len(data_k[i]['raay']) > 150 and len(data_k[i]['raay2']) > 150:
        data_k[i]['final_raay'] = data_k[i]['raay'] + "\n" + data_k[i]['raay2']
r_h = []
for i in data_h:
    if len(i['raay']) < 150 and len(i['raay2']) < 150:
        r_h.append(i)
for i in r_h:
    data_h.remove(i)
for i in range(len(data_h)):
    if len(data_h[i]['raay2']) < 150 and len(data_h[i]['raay']) > 150:
        data_h[i]['final_raay'] = data_h[i]['raay']
    elif len(data_h[i]['raay']) < 150 and len(data_h[i]['raay2']) > 150:
        data_h[i]['final_raay'] = data_h[i]['raay2']
    elif len(data_h[i]['raay']) > 150 and len(data_h[i]['raay2']) > 150:
        data_h[i]['final_raay'] = data_h[i]['raay'] + "\n" + data_h[i]['raay2']
#RAW save
dates=[]
text=[]
classs=[]
for i in data_k:
    dates.append(i['date'])
    text.append(i['final_raay'])
    classs.append("keyfari")
for i in data_h:
    dates.append(i['date'])
    text.append(i['final_raay'])
    classs.append("hoghoghi")
df = pd.DataFrame({'date':dates,'text':text,'class':classs})
df.to_excel('../data/RAW/Raw_UTF8.xlsx', index = False, header=True, encoding='utf-8')
# with open('../data/RAW/Raw.csv', 'w', newline='',encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["date", "text", "class"])
#     for i in range(len(data_k)):
#         writer.writerow([data_k[i]['date'], data_k[i]['final_raay'], "keyfari"])
#     for i in range(len(data_h)):
#         writer.writerow([data_h[i]['date'], data_h[i]['final_raay'], "hoghoghi"])

f = "./chars"
normalizer = Normalizer()
nums=["1","2","3","4","5","6","7","8","9","0"]
for k in range(len(data_k)):
    tt = data_k[k]['final_raay']
    for n in nums:
        tt = tt.replace(n," ")
    tt1 = normalizer.normalize(tt)
    tt1 = word_tokenize(tt1)
    temp=[]
    for i in range(len(tt1)):
        if len(tt1[i])<2:
            if tt1[i] != ".":
                temp.append(i)
            else:
                if len(tt1[i-1])<2 or tt1[i-1]=="الف":
                    temp.append(i)
    for ele in sorted(temp, reverse = True):
        del tt1[ele]
    out =" ".join(tt1)
    stopwords = set()
    with open(f) as file:
        stopwords.update(file.read().split())
    stopwords.remove(".")
    for j in stopwords:
        out =  out.replace(j,"")
    data_k[k]['final_raay']=out
for k in range(len(data_h)):
    tt = data_h[k]['final_raay']
    for n in nums:
        tt = tt.replace(n," ")
    tt1 = normalizer.normalize(tt)
    tt1 = word_tokenize(tt1)
    temp=[]
    for i in range(len(tt1)):
        if len(tt1[i])<2:
            if tt1[i] != ".":
                temp.append(i)
            else:
                if len(tt1[i-1])<2 or tt1[i-1]=="الف":
                    temp.append(i)
    for ele in sorted(temp, reverse = True):
        del tt1[ele]
    out =" ".join(tt1)
    stopwords = set()
    with open(f) as file:
        stopwords.update(file.read().split())
    stopwords.remove(".")
    for j in stopwords:
        out =  out.replace(j,"")
    data_h[k]['final_raay']=out
#CLEAN save
dates=[]
text=[]
classs=[]
for i in data_k:
    dates.append(i['date'])
    text.append(i['final_raay'])
    classs.append("keyfari")
for i in data_h:
    dates.append(i['date'])
    text.append(i['final_raay'])
    classs.append("hoghoghi")
df = pd.DataFrame({'date':dates,'text':text,'class':classs})
df.to_excel('../data/CLEAN/Clean_UTF8.xlsx', index = False, header=True, encoding='utf-8')
# with open('../data/CLEAN/Clean_UTF8.csv', 'w', newline='',encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["date", "text", "class"])
#     for i in range(len(data_k)):
#         writer.writerow([data_k[i]['date'], data_k[i]['final_raay'], "keyfari"])
#     for i in range(len(data_h)):
#         writer.writerow([data_h[i]['date'], data_h[i]['final_raay'], "hoghoghi"])

for k in range(len(data_k)):
    data_k[k]["sentences"] = sent_tokenize(data_k[k]["final_raay"])
for k in range(len(data_h)):
    data_h[k]["sentences"] = sent_tokenize(data_h[k]["final_raay"])
#SENTENCES save
dates=[]
sentences=[]
classs=[]
for i in data_k:
    dates.append(i['date'])
    sentences.append(i['sentences'])
    classs.append("keyfari")
for i in data_h:
    dates.append(i['date'])
    sentences.append(i['sentences'])
    classs.append("hoghoghi")
df = pd.DataFrame({'date':dates,'sentences':sentences,'class':classs})
df.to_excel('../data/SENTENCES/Sentences_UTF8.xlsx', index = False, header=True, encoding='utf-8')
# with open('../data/SENTENCES/Sentences_UTF8.csv', 'w', newline='',encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["date", "sentences", "class"])
#     for i in range(len(data_k)):
#         writer.writerow([data_k[i]['date'], data_k[i]['sentences'], "keyfari"])
#     for i in range(len(data_h)):
#         writer.writerow([data_h[i]['date'], data_h[i]['sentences'], "hoghoghi"])

for k in range(len(data_k)):
    data_k[k]["words"] = word_tokenize(data_k[k]["final_raay"])
for k in range(len(data_h)):
    data_h[k]["words"] = word_tokenize(data_h[k]["final_raay"])
#WORDS save
dates=[]
words=[]
classs=[]
for i in data_k:
    dates.append(i['date'])
    words.append(i['words'])
    classs.append("keyfari")
for i in data_h:
    dates.append(i['date'])
    words.append(i['words'])
    classs.append("hoghoghi")
df = pd.DataFrame({'date':dates,'words':words,'class':classs})
df.to_excel('../data/WORDS/Words_UTF8.xlsx', index = False, header=True, encoding='utf-8')
# with open('../data/WORDS/Words_UTF8.csv', 'w', newline='',encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["date", "words", "class"])
#     for i in range(len(data_k)):
#         writer.writerow([data_k[i]['date'], data_k[i]['words'], "keyfari"])
#     for i in range(len(data_h)):
#         writer.writerow([data_h[i]['date'], data_h[i]['words'], "hoghoghi"])

end = time.time()
print(end - start,"S")