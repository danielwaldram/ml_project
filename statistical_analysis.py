from supporting_functions import *
import csv
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.utils
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# Using https://towardsdatascience.com/building-a-one-hot-encoding-layer-with-tensorflow-f907d686bf39 as a reference for hot encoding a categorical layer
# Using https://www.youtube.com/watch?v=cJ3oqHqRBF0&ab_channel=BadriAdhikari as a reference for training a binary classifier
class OneHotEncodingLayer(layers.experimental.preprocessing.PreprocessingLayer):
    def __init__(self, vocabulary=None, depth=None, minimum=None):
        super().__init__()
        self.vectorization = layers.experimental.preprocessing.TextVectorization(output_sequence_length=1, standardize=None)

        if vocabulary:
            self.vectorization.set_vocabulary(vocabulary)
        self.depth = depth
        self.minimum = minimum

    def adapt(self, data):
        self.vectorization.adapt(data)
        vocab = self.vectorization.get_vocabulary()
        self.depth = len(vocab)
        indices = [i[0] for i in self.vectorization([[v] for v in vocab]).numpy()]
        self.minimum = min(indices)

    def call(self, inputs):
        vectorized = self.vectorization.call(inputs)
        subtracted = tf.subtract(vectorized, tf.constant([self.minimum], dtype=tf.int64))
        encoded = tf.one_hot(subtracted, self.depth)
        return layers.Reshape((self.depth,))(encoded)

    def get_config(self):
        return {'vocabulary': self.vectorization.get_vocabulary(), 'depth': self.depth, 'minimum': self.minimum}


# Reading in the set of training examples
train = []
with open("data/train_final.csv", 'r') as f:
    for line in f:
        train.append(line.strip().split(','))
columns = train[0]
# deleting first row
del train[0]
# preprocessing the data by replacing '?' with the most common value for the given attribute
train_replace_unknown = replace_unknowns(train)
# preprocessing the data by converting all categories with numbers to be binary
#train_numerical_to_binary, numerical_medians = numerical_train_data_preprocessing(train_replace_unknown)
train_str_to_flt = string_to_numerical(train_replace_unknown)
train_array = np.array(train_str_to_flt, dtype=object)
mean0 = np.mean(train_array[:, 0])
train_array[:, 0] -= mean0
std0 = np.std(train_array[:, 0])
train_array[:, 0] /= std0
mean2 = np.mean(train_array[:, 2])
train_array[:, 2] -= mean2
std2 = np.std(train_array[:, 2])
train_array[:, 2] /= std2
mean4 = np.mean(train_array[:, 4])
train_array[:, 4] -= mean4
std4 = np.std(train_array[:, 4])
train_array[:, 4] /= std4
mean10 = np.mean(train_array[:, 10])
train_array[:, 10] -= mean10
std10 = np.std(train_array[:, 10])
train_array[:, 10] /= std10
mean11 = np.mean(train_array[:, 11])
train_array[:, 11] -= mean11
std11 = np.std(train_array[:, 11])
train_array[:, 11] /= std11
mean12 = np.mean(train_array[:, 12])
train_array[:, 12] -= mean12
std12 = np.std(train_array[:, 12])
train_array[:, 12] /= std12
# Split the training data into validation and smaller training data set 20 percent of the training set is for validation
index_20percent = int(0.2*len(train_array))
np.random.shuffle(train_array)  # shuffle the dataset
XVALIDATION = train_array[:index_20percent, :-1]
YVALIDATION = train_array[:index_20percent, -1]
XTRAIN = train_array[index_20percent:, :-1]
YTRAIN = train_array[index_20percent:, -1]


# print(train_array)
# getting y-values
YTRAIN = YTRAIN.astype(np.int)
YVALIDATION = YVALIDATION.astype(np.int)
train_array_sub = XTRAIN
# train_array_sub = np.concatenate((np.transpose([XTRAIN[:, 0]]), np.transpose([XTRAIN[:, 1]]), np.transpose([XTRAIN[:, 2]]), np.transpose([XTRAIN[:, 3]])), axis=1)

train_array_y = np.transpose([YTRAIN])
validation_array_sub = XVALIDATION
#validation_array_sub = np.concatenate((np.transpose([XVALIDATION[:, 0]]), np.transpose([XVALIDATION[:, 1]]), np.transpose([XVALIDATION[:, 2]]), np.transpose([XVALIDATION[:, 3]])), axis=1)
validation_array_y = np.transpose([YVALIDATION])
# print("column 1: ", columns[1])
# print("train 1: ", np.transpose(train_array[:5, 1]))
# print("train_array_sub: ", train_array_sub)

# data frame is made up of only the 1st row which is categorical in order to build as simple a test as possible
train_df = pd.DataFrame(data=train_array_sub, columns=columns[:-1])
validation_df = pd.DataFrame(data=validation_array_sub, columns=columns[:-1])
#train_df = pd.DataFrame(data=train_array_sub, columns=[columns[0], columns[1], columns[2], columns[3]])
#validation_df = pd.DataFrame(data=validation_array_sub, columns=[columns[0], columns[1], columns[2], columns[3]])

categorical_input = layers.Input(shape=(1,), dtype=tf.string)
categorical_input_second = layers.Input(shape=(1,), dtype=tf.string)
categorical_input_3 = layers.Input(shape=(1,), dtype=tf.string)
categorical_input_4 = layers.Input(shape=(1,), dtype=tf.string)
categorical_input_5 = layers.Input(shape=(1,), dtype=tf.string)
categorical_input_6 = layers.Input(shape=(1,), dtype=tf.string)
categorical_input_7 = layers.Input(shape=(1,), dtype=tf.string)
categorical_input_8 = layers.Input(shape=(1,), dtype=tf.string)
one_hot_layer = OneHotEncodingLayer()
second_hot_layer = OneHotEncodingLayer()
third_hot_layer = OneHotEncodingLayer()
fourth_hot_layer = OneHotEncodingLayer()
fifth_hot_layer = OneHotEncodingLayer()
sixth_hot_layer = OneHotEncodingLayer()
seventh_hot_layer = OneHotEncodingLayer()
eighth_hot_layer = OneHotEncodingLayer()
one_hot_layer.adapt(train_df[columns[1]].values)
second_hot_layer.adapt(train_df[columns[3]].values)
third_hot_layer.adapt(train_df[columns[5]].values)
fourth_hot_layer.adapt(train_df[columns[6]].values)
fifth_hot_layer.adapt(train_df[columns[7]].values)
sixth_hot_layer.adapt(train_df[columns[8]].values)
seventh_hot_layer.adapt(train_df[columns[9]].values)
eighth_hot_layer.adapt(train_df[columns[13]].values)
encoded = one_hot_layer(categorical_input)
encoded_second = second_hot_layer(categorical_input_second)
encoded_third = third_hot_layer(categorical_input_3)
encoded_fourth = fourth_hot_layer(categorical_input_4)
encoded_fifth = fifth_hot_layer(categorical_input_5)
encoded_sixth = sixth_hot_layer(categorical_input_6)
encoded_seventh = seventh_hot_layer(categorical_input_7)
encoded_eighth = eighth_hot_layer(categorical_input_8)

numeric_input = layers.Input(shape=(1,), dtype=tf.float32)
numeric_input2 = layers.Input(shape=(1,), dtype=tf.float32)
numeric_input3 = layers.Input(shape=(1,), dtype=tf.float32)
numeric_input4 = layers.Input(shape=(1,), dtype=tf.float32)
numeric_input5 = layers.Input(shape=(1,), dtype=tf.float32)
numeric_input6 = layers.Input(shape=(1,), dtype=tf.float32)

concat = layers.concatenate([numeric_input, encoded, numeric_input2, encoded_second, numeric_input3, encoded_third, encoded_fourth, encoded_fifth, encoded_sixth, encoded_seventh, numeric_input4, numeric_input5, numeric_input6, encoded_eighth])

#model = models.Model(inputs=[numeric_input, categorical_input], outputs=[concat])
preprocessing_model = models.Model(inputs=[numeric_input, categorical_input, numeric_input2, categorical_input_second, numeric_input3, categorical_input_3, categorical_input_4, categorical_input_5, categorical_input_6, categorical_input_7, numeric_input4, numeric_input5, numeric_input6, categorical_input_8], outputs=concat)
#preprocessing_model = models.Model(inputs=categorical_input, outputs=encoded)
#predicted = model.predict(train_df[columns[0]], train_df[columns[1]])
#train_df[columns[0]] = pd.to_numeric(train_df[columns[0]])


# Model for making predictions
train_hot_layer_transformed = preprocessing_model.predict([train_df[columns[0]].astype(float), train_df[columns[1]], train_df[columns[2]].astype(float), train_df[columns[3]], train_df[columns[4]].astype(float), train_df[columns[5]], train_df[columns[6]], train_df[columns[7]], train_df[columns[8]], train_df[columns[9]], train_df[columns[10]].astype(float), train_df[columns[11]].astype(float), train_df[columns[12]].astype(float), train_df[columns[13]]])
validation_hot_layer_transformed = preprocessing_model.predict([validation_df[columns[0]].astype(float), validation_df[columns[1]], validation_df[columns[2]].astype(float), validation_df[columns[3]], validation_df[columns[4]].astype(float), validation_df[columns[5]], validation_df[columns[6]], validation_df[columns[7]], validation_df[columns[8]], validation_df[columns[9]], validation_df[columns[10]].astype(float), validation_df[columns[11]].astype(float), validation_df[columns[12]].astype(float), validation_df[columns[13]]])

# initialization method for the weights
initializer = tf.keras.initializers.HeNormal()  # He initialization for relu
x_initializer = tf.keras.initializers.GlorotNormal()  # xavier initialization for tanh
# depth and width values to be ran through
depths = [3, 3, 5, 5, 7, 7]
widths = [5, 7, 5, 7, 5, 7]
#depths = [3, 3, 3, 5, 5, 5]
#widths = [25, 50, 100, 25, 50, 100]
#depths = [3, 3]
#widths = [25, 50]
labels = ['width:5 depth:3', 'width:7 depth:3', 'width:5 depth:5', 'width:7 depth:5', 'width:5 depth:7', 'width:7 depth:7']
activation = 'tanh'
initializers = [initializer, x_initializer]
callback_a = ModelCheckpoint(filepath='my_best_mode.hdf5', monitor='val_loss', save_best_only=True, mode='min')
callback_b = EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1)
testing_err_tanh = []
#print("validation: ", type(validation_hot_layer_transformed))
#print("testing: ", type(train_hot_layer_transformed))
combined_hot_layer = np.append(train_hot_layer_transformed, validation_hot_layer_transformed, axis=0)
combined_y_val = np.append(train_array_y, validation_array_y, axis=0)
#print("combined: ", len(combined_hot_layer))
for val in range(len(depths)):
    testing_err_tanh.append([])
    for _ in range(10):
        # randomize the examples order
        np.random.seed()
        rand_examples = np.array([])
        rand_y_val = np.array([])
        rand_nums = np.random.choice(len(combined_hot_layer), len(combined_hot_layer), replace=False)
        rand_examples, rand_y_val = sklearn.utils.shuffle(combined_hot_layer, combined_y_val)
        #for index in range(len(rand_nums)):
        #    rand_examples = np.append(rand_examples, combined_hot_layer[rand_nums[index]])
        #    rand_y_val = np.append(rand_y_val, combined_y_val[rand_nums[index]])
        x_validation_hot_layer = rand_examples[:index_20percent, :]
        y_validation_hot_layer = rand_y_val[:index_20percent]
        x_train_hot_layer = rand_examples[index_20percent:, :]
        y_train_hot_layer = rand_y_val[index_20percent:]
        model = models.Sequential()
        model.add(layers.Dense(widths[val], input_dim=len(train_hot_layer_transformed[0, :]), activation=activation, kernel_initializer=x_initializer))
        for i in range(depths[val] - 1):
            model.add(layers.Dense(widths[val], activation=activation, kernel_initializer=x_initializer))
        model.add(layers.Dense(1, activation='sigmoid', kernel_initializer=x_initializer))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        history = model.fit(x_train_hot_layer, y_train_hot_layer, validation_data=(x_validation_hot_layer, y_validation_hot_layer), epochs=100, batch_size=10, callbacks=[callback_a, callback_b])
        # get the predictions for the model for both training and validation
        predictions = model.predict(x_validation_hot_layer)
        # get the testing and the training error
        testing_error = average_error_sign(y_validation_hot_layer, predictions)
        print('testing error: ', testing_error)
        testing_err_tanh[val].append(testing_error[0])

fig = plt.figure()
# Creating plot
plt.boxplot(testing_err_tanh, labels=labels)

# show plot
plt.show()
print(testing_err_tanh)
'''
# printing out the errors
for depth in range(len(depths)):
    print("depth " + str(depths[depth]) + ":")
    for width in range(len(widths)):
        print("     width " + str(widths[width]) + ":")
        print('         RELU testing error: ', testing_err_relu[depth*width + depth])
        print('         RELU testing error sign: ', testing_err_relu_sign[depth * width + depth])
        print('         RELU training error: ', training_err_relu[depth*width + depth])
        print('         RELU training error sign: ', training_err_relu_sign[depth * width + depth])
        print('         tanh testing error: ', testing_err_tanh[depth * width + depth])
        print('         tanh testing error sign: ', testing_err_tanh_sign[depth * width + depth])
        print('         tanh training error: ', training_err_tanh[depth * width + depth])
        print('         tanh training error sign: ', training_err_tanh_sign[depth * width + depth])


model = models.Sequential()
model.add(layers.Dense(24, input_dim=len(train_hot_layer_transformed[0, :]), activation='tanh', kernel_initializer=x_initializer))
model.add(layers.Dense(12, activation='tanh', kernel_initializer=x_initializer))
model.add(layers.Dense(6, activation='tanh', kernel_initializer=x_initializer))
#model.add(layers.Dense(120, activation='tanh', kernel_initializer=x_initializer))
#model.add(layers.Dense(5, activation='relu', kernel_initializer=x_initializer))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

callback_a = ModelCheckpoint(filepath='my_best_mode.hdf5', monitor='val_loss', save_best_only=True)
callback_b = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)

history = model.fit(train_hot_layer_transformed, train_array_y, validation_data=(validation_hot_layer_transformed, validation_array_y), epochs=200, batch_size=10, callbacks=[callback_a , callback_b])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'])
plt.show()

# predict = model.predict(train_hot_layer_transformed)
# print(predict[0:20])

# Reading in the set of training examples
test = []
with open("data/test_final.csv", 'r') as f:
    for line in f:
        test.append(line.strip().split(','))
# deleting first row
del test[0]
# deleting the first column
test = [z[1:] for z in test]
# preprocessing the data by replacing '?' with the most common value for the given attribute
test_replace_unknown = replace_unknowns(test)

# preprocessing the data by converting all categories with numbers to be binary
test_str_to_flt = string_to_numerical(test_replace_unknown)
test_array = np.array(test_str_to_flt, dtype=object)

#normalize
test_array[:, 0] -= mean0
test_array[:, 0] /= std0
test_array[:, 2] -= mean2
test_array[:, 2] /= std2
test_array[:, 4] -= mean4
test_array[:, 4] /= std4
test_array[:, 10] -= mean10
test_array[:, 10] /= std10
test_array[:, 11] -= mean11
test_array[:, 11] /= std11
test_array[:, 12] -= mean12
test_array[:, 12] /= std12
test_array_sub = test_array
#test_array_sub = np.concatenate((np.transpose([test_array[:, 0]]), np.transpose([test_array[:, 1]]), np.transpose([test_array[:, 2]]), np.transpose([test_array[:, 3]])), axis=1)
test_df = pd.DataFrame(data=test_array_sub, columns=columns[:-1])

test_hot_layer_transformed = preprocessing_model.predict([test_df[columns[0]].astype(float), test_df[columns[1]], test_df[columns[2]].astype(float), test_df[columns[3]], test_df[columns[4]].astype(float), test_df[columns[5]], test_df[columns[6]], test_df[columns[7]], test_df[columns[8]], test_df[columns[9]], test_df[columns[10]].astype(float), test_df[columns[11]].astype(float), test_df[columns[12]].astype(float), test_df[columns[13]]])

predictions = model.predict(test_hot_layer_transformed)
# print("predictions: ", predictions[0:5])

# Creating a 2D array to hold IDs and predictions
pred_2D = []
for row in range(len(predictions)):
    pred_2D.append([row + 1])

for row in range(len(predictions)):
    pred_2D[row].append(predictions[row][0])

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
'''