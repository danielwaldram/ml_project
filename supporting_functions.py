import math
import sys
import numpy as np


# average error calc for a set of examples
def average_error(y_values, predictions):
    error_sum = 0
    # run through every example
    for example in range(len(y_values)):
        error_sum += abs(y_values[example] - predictions[example])
    return error_sum/len(y_values)


# average error calc for a set of examples
def average_error_sign(y_values, predictions):
    error_sum = 0
    # run through every example
    for example in range(len(y_values)):
        if predictions[example] > 0.5:
            rounded_val = 1
        else:
            rounded_val = 0
        error_sum += abs(y_values[example] - rounded_val)
    return error_sum/len(y_values)


# takes in examples and returns the most common label among them
def most_common_label(examples):
    dict_of_labels = labels_count(examples)
    highest_count = 0
    highest_label = None
    for label in dict_of_labels:
        if dict_of_labels[label] >= highest_count:
            highest_count = dict_of_labels[label]
            highest_label = label
    return highest_label


# checks if all labels are the same and returns True if this is the case
def same_label_check(examples):
    first_label = examples[0][-1]
    for row in examples:
        if row[-1] != first_label:
            return False
    return True


# deletes all the examples with the given attribute value
def delete_examples(examples, att_val, attribute):
    new_examples = []
    for row in examples:
        if row[attribute] == att_val:
            new_examples.append(row)
    return new_examples


# Function returns a dictionary where the keys are the attributes as numbers 0 to attribute count.
#   The values of the dictionary are the possible attribute values.
def get_attributes(examples):
    dict_of_attributes = {}
    for j in range(len(examples[0]) - 1):
        dict_of_attributes[j] = []
        for i in range(len(examples)):
            if examples[i][j] not in dict_of_attributes[j]:
                dict_of_attributes[j].append(examples[i][j])
    return dict_of_attributes


# returns a dictionary with keys that are related to labels present in examples. The items are a count of
#   the number of times a given label is seen in examples
def labels_count(examples):
    # Creating a dictionary to hold each label and associated count
    dict_of_labels = {}
    for i in range(len(examples)):
        # If the dictionary item already exists, then it's total is added to
        if examples[i][-1] in dict_of_labels:
            dict_of_labels[examples[i][-1]] = dict_of_labels.get(examples[i][-1]) + 1
        # If the dictionary item doesn't already exist, then it's total starts at 1
        else:
            dict_of_labels[examples[i][-1]] = 1
    return dict_of_labels


# Takes in a decision tree and examples with a label. The function removes the labels and passes the examples
#   to the batch predictor. The predictions are then compared to the true labels to determine average error
def test_data_error_calc(decision_tree, examples):
    # first I need to get rid of the labels
    test_data_no_label = [z[:-1] for z in examples]
    test_prediction = decision_tree_batch_predictor(decision_tree, test_data_no_label)
    true_value = [z[-1] for z in examples]
    error_total = 0
    for i in range(len(examples)):
        if test_prediction[i] != true_value[i]:
            error_total += 1
    average_error = float(error_total)/len(examples)
    return average_error


# Check if a string is a number or not by trying to convert it to a float
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


# preprocessing finds the median for numerical attributes. The processed examples replaces the
#   numbers with a threshold string less than or equal to and greater than
def numerical_train_data_preprocessing(full_examples):
    # taking off the prediction
    examples = [z[:-1] for z in full_examples]
    # defining new processed examples set
    proc_ex = [z[:] for z in examples]
    # finding which of the rows are numerical rather than categorical
    numerical = {}
    # iterating through the first row of the training set to find which are numerical
    for col in range(len(examples[0])):
        if is_number(examples[0][col]):
            numerical[col] = None
    # converting numerical entries to binary threshold and creating a dictionary
    #   that holds all the numeric data
    for col in numerical.keys():
        numerical_values = []
        for row in range(len(examples)):
            numerical_values.append(int(examples[row][col]))
        median = median_calc(numerical_values)
        numerical[col] = median
        for row in range(len(examples)):
            if numerical_values[row] <= median:
                proc_ex[row][col] = "<=" + str(median)
            else:
                proc_ex[row][col] = ">" + str(median)

    # adding back in the label
    for row in range(len(proc_ex)):
        proc_ex[row].append(full_examples[row][-1])
    return proc_ex, numerical


# The numerical values need to be replaced by the binary threshold that has been determined in the training
#   set preprocessing
def numerical_test_data_preprocessing(examples, numerical_medians):
    proc_ex = [z[:] for z in examples]
    for col in numerical_medians.keys():
        for row in range(len(examples)):
            if int(examples[row][col]) <= numerical_medians[col]:
                proc_ex[row][col] = "<=" + str(numerical_medians[col])
            else:
                proc_ex[row][col] = ">" + str(numerical_medians[col])
    return proc_ex


# returns the most common value that are related to values present in the specific column of examples. The items are
#   a count of the number of times a given value is seen in examples
def most_common_value_wo_unknown(examples, col):
    # Creating a dictionary to hold each label and associated count
    dict_of_values = {}
    for i in range(len(examples)):
        # If the dictionary item already exists, then it's total is added to
        if examples[i][col] in dict_of_values:
            dict_of_values[examples[i][col]] = dict_of_values.get(examples[i][col]) + 1
        # If the dictionary item doesn't already exist, then it's total starts at 1
        elif examples[i][col] == 'unknown':
            continue
        else:
            dict_of_values[examples[i][col]] = 1

    highest_count = 0
    highest_value = None
    for value in dict_of_values:
        if dict_of_values[value] >= highest_count:
            highest_count = dict_of_values[value]
            highest_value = value
    return highest_value


# This function will replace values listed as unknown with the most common label for the given attribute
def replace_unknowns(examples):
    # find the most common value for each column corresponding to an attribute
    most_common_values = []
    for col in range(len(examples[0])):
        most_common_values.append(most_common_value_wo_unknown(examples, col))

    # replace unknowns in examples
    proc_ex = [z[:] for z in examples]
    for row in range(len(examples)):
        for col in range(len(examples[row])):
            if examples[row][col] == '?':
                proc_ex[row][col] = most_common_values[col]
    return proc_ex


# preprocessing finds the median for numerical attributes. The processed examples replaces the
#   numbers with a threshold string less than or equal to and greater than
def string_to_numerical(full_examples):
    # taking off the prediction
    examples = [z[:-1] for z in full_examples]
    # defining new processed examples set
    proc_ex = [z[:] for z in examples]
    # finding which of the rows are numerical rather than categorical
    numerical = {}
    # iterating through the first row of the training set to find which are numerical
    for col in range(len(examples[0])):
        if is_number(examples[0][col]):
            numerical[col] = None
    # converting numerical entries to binary threshold and creating a dictionary
    #   that holds all the numeric data
    for col in numerical.keys():
        for row in range(len(examples)):
            proc_ex[row][col] = int(proc_ex[row][col])
    # adding back in the label
    for row in range(len(proc_ex)):
        proc_ex[row].append(full_examples[row][-1])
    return proc_ex
