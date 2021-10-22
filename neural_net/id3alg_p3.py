import math
import sys


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


# This is the main ID3 tree building algorithm
def id3(prev_node_examples, node_examples, full_attributes, remain_attributes, max_size, gain_type):
    # id3 will return this node represented as a dictionary
    node = {}
    # If there are no examples left. The examples from the previous node are used to determine the leaf's label
    if len(node_examples) == 0:
        # node is labeled a leaf
        node['subnodes'] = 'leaf'
        # leaf label is made most common from the previous nodes examples
        node['label'] = most_common_label(prev_node_examples)
        return node

    # Check if all the examples have the same label
    if same_label_check(node_examples):
        node['subnodes'] = 'leaf'
        # they're all the same, so I will make the first examples label the label for the leaf node
        node['label'] = node_examples[0][-1]
        return node

    # Check if there are any more attributes on which to split
    if len(remain_attributes.keys()) == 0:
        node['subnodes'] = 'leaf'
        # they're all the same, so I will make the first examples label the label for the leaf node
        node['label'] = most_common_label(node_examples)
        return node

    # Limit the size of the tree to the max size
    if max_size is not None:
        if (len(full_attributes) - len(remain_attributes)) == max_size:
            node['subnodes'] = 'leaf'
            # they're all the same, so I will make the first examples label the label for the leaf node
            node['label'] = most_common_label(node_examples)
            return node

    # Need to find the attribute that will best split the data out of all the remaining attributes
    if gain_type == 'entropy':
        best_attribute, best_gain = highest_info_gain_entropy(node_examples, remain_attributes)
    elif gain_type == 'me':
        best_attribute, best_gain = highest_info_gain_me(node_examples, remain_attributes)
    elif gain_type == 'gini':
        best_attribute, best_gain = highest_info_gain_gini(node_examples, remain_attributes)
    else:
        sys.exit("You entered an incorrect string for gain_type.")

    # Now know the attribute for the next split, so this can be saved along with the gain
    node['attribute'] = best_attribute
    node['subnodes'] = {}

    # need to remove the attribute from the remaining attributes
    new_remain_attributes = remain_attributes.copy()
    del new_remain_attributes[best_attribute]

    for attribute_val in full_attributes[best_attribute]:
        # need to delete any examples that don't fit this attribute value
        new_node_examples = delete_examples(node_examples, attribute_val, best_attribute)
        new_prev_node_examples = [x[:] for x in node_examples]
        node['subnodes'][attribute_val] = id3(new_prev_node_examples, new_node_examples, full_attributes, new_remain_attributes, max_size, gain_type)

    return node


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


# Function that calculates the entropy of a given a set of examples. This function only
#   uses the final column of the array, which holds the label
def entropy_calc(examples):
    # get dictionary of labels
    dict_of_labels = labels_count(examples)

    # Calculating the entropy of the set by iterating through the dictionary adding up entropy contributions
    entropy = 0
    for key in dict_of_labels.keys():
        entropy = entropy - (float(dict_of_labels[key])/len(examples))*math.log(float(dict_of_labels[key])/len(examples), 2)
    return entropy


# Function that calculates the majority error of a given a set of examples. This function only
#   uses the final column of the array, which holds the label
def majority_error_calc(examples):
    # find the most common label
    label_most = most_common_label(examples)
    if label_most is None:
        return 0
    # get counts for every label in dictionary
    label_dict = labels_count(examples)
    # calculate the majority error
    maj_err = (len(examples) - label_dict[label_most])/float(len(examples))
    # return
    return maj_err


# Function that calculates the gini index of a given a set of examples. This function only
#   uses the final column of the array, which holds the label
def gini_index_calc(examples):
    # get dictionary of labels
    dict_of_labels = labels_count(examples)

    # Calculating the gini index of the set by iterating through the dictionary adding up probability contributions
    gi_sum = 0
    for key in dict_of_labels.keys():
        gi_sum = gi_sum + pow(float(dict_of_labels[key]) / len(examples), 2)
    gi = 1 - gi_sum
    return gi


# Function returns the highest information gain using ME method. Using all the given examples and all the given
#   attributes to determine which attribute splits the best
def highest_info_gain_gini(examples, attributes):
    # ME of the set of examples
    set_gi = gini_index_calc(examples)
    # dictionary holding all the information gain values
    information_gains = {}
    # highest information gain of the given attributes
    highest_info_gain = 0
    # highest information gain key
    highest_info_gain_key = list(attributes)[0]
    # iterating through each attribute provided
    for key in attributes.keys():
        # info gain for specific attribute value
        info_gain_sum = 0
        # split the example by attribute value
        sub_examples = attribute_split(examples, key, attributes)
        # iterating through every row (corresponding to an attribute value) of sub_examples to get contribution to gain
        for row in sub_examples:
            info_gain_sum = info_gain_sum + float(len(sub_examples[row])) / len(examples) * gini_index_calc(
                sub_examples[row])
        # Calculating the total information gain
        information_gains[key] = set_gi - info_gain_sum
        if information_gains[key] >= highest_info_gain:
            highest_info_gain_key = key
            highest_info_gain = information_gains[key]
    return highest_info_gain_key, highest_info_gain


# Function returns the highest information gain using ME method. Using all the given examples and all the given
#   attributes to determine which attribute splits the best
def highest_info_gain_me(examples, attributes):
    # ME of the set of examples
    set_me = majority_error_calc(examples)
    # dictionary holding all the information gain values
    information_gains = {}
    # highest information gain of the given attributes
    highest_info_gain = 0
    # highest information gain key
    highest_info_gain_key = list(attributes)[0]
    # iterating through each attribute provided
    for key in attributes.keys():
        # info gain for specific attribute value
        info_gain_sum = 0
        # split the example by attribute value
        sub_examples = attribute_split(examples, key, attributes)
        # iterating through every row (corresponding to an attribute value) of sub_examples to get contribution to gain
        for row in sub_examples:
            info_gain_sum = info_gain_sum + float(len(sub_examples[row])) / len(examples) * majority_error_calc(
                sub_examples[row])
        # Calculating the total information gain
        information_gains[key] = set_me - info_gain_sum
        if information_gains[key] >= highest_info_gain:
            highest_info_gain_key = key
            highest_info_gain = information_gains[key]
    return highest_info_gain_key, highest_info_gain


# Function returns the highest information gain using entropy method. Using all the given examples and all the given
#   attributes to determine which attribute splits the best
def highest_info_gain_entropy(examples, attributes):
    # entropy of the set of examples
    set_entropy = entropy_calc(examples)
    # dictionary holding all the information gain values
    information_gains = {}
    # highest information gain of the given attributes
    highest_info_gain = 0
    # highest information gain key
    highest_info_gain_key = list(attributes)[0]
    # iterating through each attribute provided
    for key in attributes.keys():
        # info gain for specific attribute value
        info_gain_sum = 0
        # split the example by attribute value
        sub_examples = attribute_split(examples, key, attributes)
        # iterating through every row (corresponding to an attribute value) of sub_examples to get contribution to gain
        for row in sub_examples:
            info_gain_sum = info_gain_sum + float(len(sub_examples[row]))/len(examples)*entropy_calc(sub_examples[row])
        # Calculating the total information gain
        information_gains[key] = set_entropy - info_gain_sum

        if information_gains[key] >= highest_info_gain:
            highest_info_gain_key = key
            highest_info_gain = information_gains[key]
    return highest_info_gain_key, highest_info_gain


# Function to split the examples up by attribute value. Returns a dictionary with an entry for every attribute value and
#   a list for all the rows under it.
def attribute_split(examples, key, attributes):
    # dictionary to hold all the example labels split up by the value of the current attribute of interest
    sub_examples = {}
    # temp_examples is the examples list but it gets smaller as its looped through to improve efficiency
    temp_examples = examples[:][:]
    # iterating through the list of possible values under the given attribute
    for k in range(len(attributes[key])):
        # a new row in sub_examples is saved for the current attribute value
        sub_examples[attributes[key][k]] = []
        # index for the temporary examples array
        j = 0
        for i in range(len(temp_examples)):
            # if the example has the same attribute value as the current one of interest, then the examples label
            #   is added to its row of the sub_examples array. If not then the temporary examples index "j" is
            #   increased.
            if temp_examples[j][key] == attributes[key][k]:
                sub_examples[attributes[key][k]].append(temp_examples[j][-1])
                temp_examples.remove(temp_examples[j])
            else:
                j = j + 1
    return sub_examples


# This function will take in a decision_tree and an example without a label.
#   the tree then makes a prediction on what the label of the example is
def decision_tree_predictor(decision_tree, example):
    if 'attribute' in decision_tree:
        # determine what attribute is split by this tree
        attribute = decision_tree["attribute"]
        # the attribute number should correspond to the index number in the example
        value = example[attribute]
        # finding the branch specific to the value for the attribute
        value_branch = decision_tree['subnodes'][value]
        label = decision_tree_predictor(value_branch, example)
        return label
    return decision_tree['label']


# This function will take in a decision tree and a set of examples with no labels and return a list of label predictions
def decision_tree_batch_predictor(decision_tree, examples):
    prediction = []
    for row in examples:
        prediction.append(decision_tree_predictor(decision_tree, row))
    return prediction


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


# Calculate the median of a set of data
def median_calc(numeric_list):
    ordered_list = sorted(numeric_list)
    # check if the list is even or odd
    if len(numeric_list) % 2 == 0:
        median = (ordered_list[int(len(numeric_list)/2)] + ordered_list[int(len(numeric_list)/2) - 1])/float(2)
        return median
    else:
        median = (ordered_list[int(len(numeric_list)/2)])
        return median


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