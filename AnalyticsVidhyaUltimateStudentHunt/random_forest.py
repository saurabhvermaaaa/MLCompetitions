import pandas
import numpy
from sklearn.ensemble import RandomForestRegressor as forest
from sklearn import preprocessing
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor as boosting
from sklearn.model_selection import GridSearchCV

columns = ["Park_ID","Date","Direction_Of_Wind","Average_Breeze_Speed","Average_Atmospheric_Pressure","Max_Atmospheric_Pressure","Min_Atmospheric_Pressure","Min_Ambient_Pollution","Max_Ambient_Pollution","Average_Moisture_In_Park","Max_Moisture_In_Park","Min_Moisture_In_Park"]

# columns = ["Park_ID","Date","Direction_Of_Wind","Average_Breeze_Speed","Max_Breeze_Speed","Min_Breeze_Speed","Average_Moisture_In_Park","Max_Moisture_In_Park","Min_Moisture_In_Park"]

def preprocess(data):
    data["Date"] = [float(datetime.strptime(x, '%d-%m-%Y').strftime("%s")) % (86400 * 365) for x in data['Date']]
    data = data.interpolate()
    data = data[columns]
    data = preprocessing.scale(data)
    return data


data = pandas.read_csv("./train.csv", na_values=[''])
test = pandas.read_csv("./test.csv", na_values=[''])

test_ids = test["ID"]
train_out = numpy.concatenate(data.as_matrix(['Footfall']))

train_in = preprocess(data)
test_in = preprocess(test)

boost = boosting(loss='lad', learning_rate=0.05, n_estimators=100, subsample=0.8, criterion='mse', min_samples_split=800, min_samples_leaf=140, min_weight_fraction_leaf=0.0, max_depth=9, min_impurity_split=1e-07, init=None, random_state=500, max_features=8, alpha=0.9, verbose=10, max_leaf_nodes=None, warm_start=True, presort='auto')

# param_grid = {}

# search = GridSearchCV(boost, param_grid, scoring=None, fit_params=None, n_jobs=2, iid=True, refit=True, cv=None, verbose=10, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)

# search.fit(train_in, train_out)

# print search.cv_results_, search.best_params_
# print search.best_score_

boost.fit(train_in, train_out)

# from feature_selection import plot_importances

# plot_importances(boost, columns, train_in)

with open("submission.csv", "w") as OUT:
	OUT.write("ID,Footfall\n")
	for x, y in zip(test_ids, boost.predict(test_in)):
		OUT.write(str(x) + "," + str(int(y)) + "\n")
