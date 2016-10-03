import numpy
import matplotlib as plt
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

def plot_importances(clf,columns, train_in):
    importances = clf.feature_importances_
    indices = numpy.argsort(importances)[::-1]

    print("Feature ranking:")

    for f in range(train_in.shape[1]):
        print("%d. %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))

    # print columns

    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(train_in.shape[1]), importances[indices], color="b", align="center")
    # plt.xticks(range(train_in.shape[1]), indices)
    # plt.xlim([-1, train_in.shape[1]])
    # plt.show()

def remove_low_variance_features(train_in):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit_transform(train_in)

def descrive(data):
    print data.info()
    print data.describe()


