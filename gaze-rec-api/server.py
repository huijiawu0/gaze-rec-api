import subprocess
import pandas as pd
import numpy as np
import faiss

from flask import Flask, request, jsonify

app = Flask(__name__)

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
print("Init news done!")


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/api/rec", methods=['POST'])
def get_recommend():
    content = request.json
    print(content)
    
    ret = {}
    result = []
    try:
        items = content['data']['item']
        rank_threshold = content['config']['rank_param']['threshold']
        rank_top_k = content['config']['rank_param']['top_k']
        try:
            text = items[0]
            vec = bc.encode([text])
            D, I = index.search(vec, rank_top_k)
            for d, i in zip(D.flatten(), I.flatten()):
                result.append({"item": news['headline'][i], "value": d.item()})
            if len(result) == rank_top_k:
                ret['status'] = 200
            else:
                ret['status'] = 201
            ret['data'] = result
            return jsonify(ret)
        except IndexError:
            return "1002"
    except KeyError:
        return "1001"


if __name__ == "__main__":
    app.run(port=8000, debug=True)