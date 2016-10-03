import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
import pandas
import numpy

encoder = preprocessing.LabelEncoder()

def preprocess(data):
	data = data.fillna(-1.0)
	data['Date'] =[float (datetime.strptime(x, '%d-%m-%Y').strftime("%s")) for x in data['Date']]
	return data


data = pandas.read_csv("./train.csv", na_values=[''])
test = pandas.read_csv("./test.csv", na_values=[''])

data = preprocess(data)
test = preprocess(test)
columns = ["ID","Park_ID","Date","Direction_Of_Wind","Average_Breeze_Speed","Max_Breeze_Speed","Min_Breeze_Speed","Var1","Average_Atmospheric_Pressure","Max_Atmospheric_Pressure","Min_Atmospheric_Pressure","Min_Ambient_Pollution","Max_Ambient_Pollution","Average_Moisture_In_Park","Max_Moisture_In_Park","Min_Moisture_In_Park","Location_Type","Footfall"]
for col in columns:
	plt.scatter(data[col], data['Footfall'])
	plt.xlabel(col)
	plt.ylabel('Footfall')
	plt.show()

