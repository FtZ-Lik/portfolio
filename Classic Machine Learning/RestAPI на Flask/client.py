# -*- coding: utf-8 -*-
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

# формируем запрос
def send_json(input):
    body = dict(input)
    myurl = 'http://127.0.0.1:1123/predict'
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(myurl, json=body, headers=headers)
    return response.json()

client_df = pd.read_csv("./train_case2.csv", sep=';')
X_train, X_test, y_train, y_test = train_test_split(client_df.drop(['cardio', 'id'], axis=1), client_df['cardio'], random_state=42)

response = send_json(X_test.iloc[21])
print(f'Предсказание класса: {int(response["pred"])} ({response["pred_proba"]})')