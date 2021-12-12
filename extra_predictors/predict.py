
# import libraries
from flask import Flask
from flask import request
from flask import jsonify

import numpy as np
import pandas as pd
import pickle 

import json

#load models
with open("../pipeline.bin", 'rb') as f: pipeline = pickle.load(f)

#config app Flask
app = Flask('TNPrice')

@app.route('/predict', methods=['POST']) #/predict as gateway
def predict():
    data = request.get_json() #get json data as dict
    data = json.loads(data);
    print(data) #print them (comment me)

    #df = data;
    df = pd.DataFrame.from_records(data);
    #print(df)
    y_pred = pipeline.predict(df) #predict 
    print(y_pred)
    
    result = {
        'log_price': float(y_pred)
    } #only one result, as we are using decision tree

    return jsonify(result) #convert dict to json string and serve them


if __name__ == "__main__":    app.run(debug=True, host='0.0.0.0', port=9696) #serve in local host via port 9696, with verbose output