#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
#print number of entries
print "Total entries:", len(enron_data)

#get number of features for each entry
p1 =  enron_data["LAY KENNETH L"]
print "Features per entry:", len(p1)

## print the list of items in the data
# for i, v in enumerate(enron_data):
#      print i, v, len(v)

## calculate the number of pois
pois = 0
for i in enron_data:
    if enron_data[i]["poi"] == True:
        pois += 1

print "POIs:", pois

## get total stock value for person
jprentice = enron_data["PRENTICE JAMES"]["total_stock_value"]
print "James Prentice stock value:", jprentice

## get emails from  Wesley Colwell to persons of interest
emails = enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "Wesley Colwell to persons of interest:", emails

## find value of stock options exercised by Jeffrey K Skilling
jskilling = enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print "Jeffrey K Skilling stock options:", jskilling

##
lay = enron_data["LAY KENNETH L"]["total_payments"]
print "Kenneth Lay total value:", lay

skilling = enron_data["SKILLING JEFFREY K"]["total_payments"]
print "Jeffrey K Skilling total value:", skilling

fastow = enron_data["FASTOW ANDREW S"]["total_payments"]
print "Andrew s Fastow total value:", fastow

## calculate the people with salaries and emails
salaries = 0
emails = 0
for i in enron_data:
    if enron_data[i]["salary"] != 'NaN':
        salaries += 1
    if enron_data[i]["email_address"] != 'NaN':
        emails += 1


print "salaries:", salaries
print "emails:", emails

## calculate people without total payments
payments = 0
for i in enron_data:
    if enron_data[i]["total_payments"] == 'NaN':
        payments += 1

print "# of persons with no payments:", payments
print "% of no payments:", payments/ (len(enron_data) * 1.0)