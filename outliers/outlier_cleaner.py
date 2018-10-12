#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
   
    ### your code goes here
   
    ## calculate the residual error (prediction - net worth) for each element
    residual_errors = []
    for idx, val in enumerate(predictions):
        residual_errors.append(predictions[idx] - net_worths[idx])

    ## combine all data points into tuple and sort by residual error
    data = zip(ages, net_worths, predictions, residual_errors)
    data.sort(key=lambda tup: tup[3]) #sort by 4th element of tuple
    data = data[:-9] #drop the last 9 elements
    # print "count = ", len(data) #check we have 81 elements

    ## remove the residual error from the list
    for i, var in enumerate(data):
        newTuple = (var[0], var[1], var[2])
        cleaned_data.append(newTuple) 

    # print cleaned_data
    return cleaned_data

