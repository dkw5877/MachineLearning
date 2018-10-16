#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""
import sys
sys.path.append("../tools/")
import pickle
import numpy

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)
# print "data = ", data_dict

## get the dictionary for each employee
contents = data_dict.values()

## for each employee data extract the required data for the key (exercised stock options or salary) into an array
salaries = [d['salary'] for d in contents]
stock_options = [d['exercised_stock_options'] for d in contents]

### filter out 'NaN' from the array of values
salaries = [x for x in salaries if str(x) != 'NaN']
stock_options = [x for x in stock_options if str(x) != 'NaN']

# print min and max values from the array
minValue = min(salaries)
print "min salary =",minValue
maxValue = max(salaries)
print "max salary =",maxValue

minValue = min(stock_options)
print "min options =",minValue
maxValue = max(stock_options)
print "max options =",maxValue

def createMatrix(arr):
    import array
    output = []
    for i in arr:
        output.append([float(i)])
    return output

##scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
## rescale the salary data
salaries_matrix = createMatrix(sorted(salaries))
scaled_salaries = scaler.fit_transform(salaries_matrix)
print scaled_salaries

## rescale the exercised_stock_options data
options_matrix = createMatrix(sorted(stock_options))
scaled_options = scaler.fit_transform(options_matrix)

### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
### we use underscore to include feature_3 in the loop, but ignore it for the scatter plot
for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(finance_features)
pred = kmeans.predict(finance_features)


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
