#!/usr/bin/python

import sys
sys.path.append("../tools/")
import numpy
import pickle
import random

from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop( "TOTAL", 0 )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
## NOTE: we need to add this line before graphing to avoid ImportError: No module named functools_lru_cache
from sklearn.linear_model import LinearRegression
# reg = LinearRegression()
# reg.fit(ages_train, net_worths_train)

# print "r-square:  ", reg.score(ages_test, net_worths_test), "(test data)"
# print "r-square:  ", reg.score(ages_train, net_worths_train)
# print "slope:     ", reg.coef_
# print "intercept: ", reg.intercept_


#### visualization #####
import matplotlib.pyplot as plt

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()



