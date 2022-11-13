import requests
res = requests.post('http://localhost:8000/api/rec', json={
    "data": {
        "user": 0,
        "item": ["长征胜利红色旅游系列活动启动"],
        "context": ['gaze_feat1', 'gaze_feat2', 'gaze_feat3']
    },
    "controller": {
        "recall_mode": 0,
        "rank_mode": 0,
        "rerank_mode": 0
    },
    "config": {
        "recall_param": {
            "threshold": 1.0,
            "top_k": 50
        },
        "rank_param": {
            "threshold": 1.0,
            "top_k": 2
        }
    },
    "debug": 0
}
)
if res.ok:
    print(res.json())