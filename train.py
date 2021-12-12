"""####Train###########


"""

# libarires
import numpy as np
import pandas as pd

#splitter
from sklearn.model_selection import train_test_split

#pipeline
from sklearn.pipeline import Pipeline

#transformers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


#model
from sklearn.ensemble import GradientBoostingRegressor

#save pickle
import pickle

#read database
df = pd.read_csv("Property Prices in Tunisia.csv") #load file

#translation
df.replace(df["category"].unique(), ['Land and Farms', 'Apartments', 'Holiday rentals',
       'Shops, Businesses and Industrial Premises', 'Houses and Villas',
       'Flatshare', 'Offices and Trays'],inplace=True)

df.replace(df["type"].unique(), ['For sale', 'For rent'],inplace=True)

df.replace("Autres villes","Others", inplace=True)

del df["price"] 


# data splitting

#split according to random state = 1
df_trainval, df_test = train_test_split(df, test_size=0.2, random_state=1) #split train+val [80%], test [20%]
df_train, df_val = train_test_split(df_trainval, test_size=0.25, random_state=1) #split train[60%] val [20%]

#reset, drop index
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_test = df_test["log_price"].values
y_train = df_train["log_price"].values
y_val = df_val["log_price"].values

del df_test["log_price"]
del df_train["log_price"]
del df_val["log_price"]



# feature types
feat_cat = ['category','type','city','region']
feat_num = ['room_count','bathroom_count','size']

# transformations
transformations = [
    ('Scal_num', RobustScaler() , feat_num),
    ('categorical', OneHotEncoder(dtype=np.int32,handle_unknown = 'ignore'), feat_cat)
]
transf = ColumnTransformer(transformations, remainder='drop')

#Gradient Boost Regressor
pipeline = Pipeline([
    ('transformer', transf),
    ('gb', GradientBoostingRegressor(max_depth=1000,learning_rate=0.03))
])

pipeline.fit(df_train, y_train)

with open("pipeline.bin", 'wb') as f:  pickle.dump(pipeline,f)