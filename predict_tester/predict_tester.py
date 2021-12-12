
import requests
import json
import pickle

with open("df_test.bin", 'rb') as f: df_test = pickle.load(f)
with open("y_test.bin", 'rb') as f: y_test = pickle.load(f)


while True:
    id = input("Give me an ID from the dataset:")
    try:
        id = int(id)
        example = df_test.iloc[id:id+1,:].to_json()
        expected_res = y_test[id]



        url = "http://127.0.0.1:9696/predict"
        results = requests.post(url, json=example).json()
        print(df_test.iloc[id:id+1,:])
        print("Price for ID %d : %f" % (id, results["log_price"]))

        print("Database results: %f" % expected_res)
    except Exception as e:
        print(e)