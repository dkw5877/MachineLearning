#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)



### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"

### use the included data files, instead of the ones from (Lesson 10) mini-project
# words_file = "word_data_overfit.pkl" 
# authors_file = "email_authors_overfit.pkl" 

word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

print "training points: ", len(features_train)

### your code goes here
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

t0 = time()
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

##predictions
t0 = time()
predictions = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

##accuracy
accuracy = accuracy_score(predictions, labels_test)
print "accuracy = ", accuracy, "\n"
## .94 using our data 
## .95 using provided data
## answer is ~0.95 (so either will work)

## get the most important feature
## this produces the correct value but the wrong index, not sure how that is possible
## correct answer is index 33614, val 0.764705882353
## Note: correct number of features for lesson 10 was 38757, while our data got 38755
## Note: my data set gives index 33613, while the included data set gives index 33604
## PART II - remove the  word sshacklensf and re-run
## correct answer is cgermannsf at index 14343.
## Note: my data set gives index 14343
top_feature = max(clf.feature_importances_)
index_feature = numpy.argmax(clf.feature_importances_)
print "top_feature: ", top_feature
print "index_feature: ", index_feature, "\n"

## try iterating the array to show the index and values
for idx, val in enumerate(clf.feature_importances_):
    if val > 0.02:
        print "idx: ", idx, "val:", float(val)


## Do TfIdf vectorization 
## number of feature between our data (38753) and provided data (38747) varies
## answer is sshacklensf and found at index 34409 in our data set
## answer is sshacklensf and found at index 34400 in the provided data set
## Note: the following above indices are only accurate as long as the answer (sshacklensf) is included in the data set
## PART II - remove the  word sshacklensf and re-run
## answer is cgermannsf and found at index 14550 in the our data set
vectorizer.fit_transform(word_data)
features = vectorizer.get_feature_names()
print "features len:", len(features)
print "important feature: ", features[14550]

# print features to file so we can find the index manually
# import sys
# sys.stdout = open("our_revised_data.txt", "w+")
# for f in features:
#     print f








