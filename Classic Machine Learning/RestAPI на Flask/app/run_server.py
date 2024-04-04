#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:02:45 2023

@author: ftz
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import dill

with open('./models/GradientBoostingClassifier_cardio.dill', 'rb') as in_strm:
    #в файле run_server.py путь изменен на './models/GradientBoostingClassifier_cardio.dill'
    model = dill.load(in_strm)
    
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Обработчики и запуск Flask
app = Flask(__name__)
#run_with_ngrok(app)  # Start ngrok when app is run

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.route('/exit', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route("/", methods=["GET"])
def general():
    return "Welcome to prediction process\n"

@app.route('/predict', methods=['POST'])
def predict():
    data = {"success": False}

    request_json = request.get_json()

    feature_dict = {}
    
    for feature in features:
        if feature in request_json.keys() :
            feature_dict[feature] = [request_json[feature]]
        else:
            feature_dict[feature] = np.nan


    #print(feature_dict)
    pred_proba = model.predict_proba(pd.DataFrame(feature_dict, index=[0]))
    pred = model.predict(pd.DataFrame(feature_dict, index=[0]))
    data["pred_proba"] = pred_proba[:, 1][0]
    data["pred"] = float(pred[0])
        # indicate that the request was a success
    data["success"] = True
    print('OK')
    #print(data)

        # return the data dictionary as a JSON response
    return jsonify(data)


if __name__ == '__main__':
    app.run(port=1123)