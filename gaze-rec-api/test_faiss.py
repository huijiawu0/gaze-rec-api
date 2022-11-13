import json
import subprocess

import pandas as pd
import numpy as np
import faiss

bert_command = 'bert-serving-start -model_dir /Users/hjw/Downloads/chinese_L-12_H-768_A-12'
process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)
from bert_serving.client import BertClient
bc = BertClient()
print("Init BertClient done!")

news_vec = pd.read_pickle('/Users/hjw/Downloads/news_vec.pkl')
news_vec = np.asarray(news_vec)
d = len(news_vec[0])
index = faiss.IndexFlatL2(d)
index.add(news_vec)
print("Init faiss index done!")

news = pd.read_csv('/Users/hjw/WebstormProjects/eyecontrol/eyectr1/chinese_news.csv')
text = "长征胜利红色旅游系列活动启动"
vec = bc.encode([text])
rank_top_k = 3
D, I = index.search(vec, rank_top_k)
result = []
for d, i in zip(D.flatten(), I.flatten()):
	result.append({"item": news['headline'][i], "value": d})
print(json.dumps(result))