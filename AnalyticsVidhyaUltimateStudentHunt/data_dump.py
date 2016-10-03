import pandas
import numpy
from sklearn.ensemble import RandomForestRegressor as forest
from sklearn import preprocessing
from datetime import datetime
from sklearn import cross_validation
import matplotlib.pyplot as plt
import pickle

all_columns = ["ID","Park_ID","Date","Direction_Of_Wind","Average_Breeze_Speed","Max_Breeze_Speed","Min_Breeze_Speed","Var1","Average_Atmospheric_Pressure","Max_Atmospheric_Pressure","Min_Atmospheric_Pressure","Min_Ambient_Pollution","Max_Ambient_Pollution","Average_Moisture_In_Park","Max_Moisture_In_Park","Min_Moisture_In_Park","Location_Type","Footfall"]

columns = ["Min_Moisture_In_Park", "Date", "Direction_Of_Wind", "Var1", "Average_Moisture_In_Park", "Average_Breeze_Speed", "Max_Moisture_In_Park", "Park_ID"]

encoder = preprocessing.LabelEncoder()

def preprocess(data):
    data = data.fillna(-1.0)
    data['Date'] =[float (datetime.strptime(x, '%d-%m-%Y').strftime("%s")) for x in data['Date']]
    return data


data = pandas.read_csv("./train.csv", na_values=[''])
test = pandas.read_csv("./test.csv", na_values=[''])

data = preprocess(data)
test = preprocess(test)

train_in = data.as_matrix(columns)
test_in = test.as_matrix(columns)
test_ids = test["ID"]
train_out = numpy.concatenate(data.as_matrix(['Footfall']))

save = {
    "train_in":train_in, 
    "test_in":test_in, 
    "test_ids":test_ids, 
    "train_out":train_out
}

pickle.dump(save)
