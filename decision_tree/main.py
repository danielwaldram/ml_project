from id3alg_p3 import *
import csv
from matplotlib import pyplot as plt

# Reading in the set of training examples
train = []
with open("data/train_final.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))
# deleting first row
del train[0]

# preprocessing the data by replacing '?' with the most common value for the given attribute
train_replace_unknown = replace_unknowns(train)

# preprocessing the data by converting all categories with numbers to be binary
train_numerical_to_binary, numerical_medians = numerical_train_data_preprocessing(train_replace_unknown)
# I may have made a mistake because my prediction column also converted to binary, which I thought was not supposed to happen

# getting all the attributes that are in examples
full_attributes = get_attributes(train_numerical_to_binary)
# remaining attributes is passed to id3 because it will change on recursion, it will remain the same
#   for this initial call
remain_attributes = full_attributes.copy()

index_20percent = int(0.2*len(train_numerical_to_binary))
TRAIN = train_numerical_to_binary[index_20percent:]
VALIDATION = train_numerical_to_binary[:index_20percent]

error = []
validation_error = []
for i in range(1, 15):
    print("TREE SIZE: ", i)
    dec_tree_entropy = id3(TRAIN, TRAIN, full_attributes, remain_attributes, i, "entropy")
    error.append(test_data_error_calc(dec_tree_entropy, TRAIN))
    validation_error.append(test_data_error_calc(dec_tree_entropy, VALIDATION))
print("training error: ", error)
print("validation error: ", validation_error)

plt.plot(error)
plt.plot(validation_error)
plt.ylabel('error')
plt.xlabel('tree size')
plt.legend(['training data', 'validation data'])
plt.show()


# train with entropy
dec_tree_entropy = id3(train_numerical_to_binary, train_numerical_to_binary, full_attributes, remain_attributes, 5, "entropy")
error = test_data_error_calc(dec_tree_entropy, train_numerical_to_binary)




# read in the test data
test = []
with open("data/test_final.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))
# remove header
del test[0]
# remove first column
test = [z[1:] for z in test]

# replace unknown test values
processed_wo_unk_test = replace_unknowns(test)
# Change numerical to binary
processed_numerical_to_binary_test = numerical_test_data_preprocessing(processed_wo_unk_test, numerical_medians)

# predictions holds the predictions for the test dataset
predictions = decision_tree_batch_predictor(dec_tree_entropy, processed_numerical_to_binary_test)
# Creating a 2D array to hold IDs and predictions
pred_2D = []
for row in range(len(predictions)):
    pred_2D.append([row + 1])

for row in range(len(predictions)):
    pred_2D[row].append(predictions[row])

# Write predictions to the csv file
with open('data/predictions.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write the header
    header = ["ID", "Prediction"]
    writer.writerow(header)
    # write a row to the csv file
    for row in pred_2D:
        writer.writerow(row)
