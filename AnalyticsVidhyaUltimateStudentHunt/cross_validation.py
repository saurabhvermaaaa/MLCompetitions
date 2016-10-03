import pandas
import numpy
from sklearn.ensemble import RandomForestRegressor as forest
from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_in, train_out, test_size=0.2, random_state=0)

clf = forest(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=500, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=10, warm_start=False)

clf.fit(X_train, y_train)

print clf.score(X_test, y_test)

search = GridSearchCV(boost, param_grid, scoring=None, fit_params=None, n_jobs=2, iid=True, refit=True, cv=None, verbose=10, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)

search.fit(train_in, train_out)

print search.cv_results_, search.best_params_, search.best_score_
