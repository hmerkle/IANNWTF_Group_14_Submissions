# imports
import tensorflow as tf
import pandas as pd

# own imports
from util_functs import *
from model import *

#Preparation of data

# load data into a data frame
df_wine = pd.read_csv(r"C:\Users\hanna\Dokumente\Uni_classes\ANNsTensorflow\homework\H4\winequality-red.csv", sep = ";", index_col = False)

# set a qualoity threshold
### median of qualities not efficient so the threshold was fixed
threshold = 7

# prepare all data
### prepare dataframe and split into train, test, validate
### convert dataframes into tensorflow datasets
### prepare datasets
train_dataset, test_dataset, val_dataset = prepare_all_data(df_wine, threshold)  


# Training

### Hyperparameters
num_epochs = 10
learning_rate = 0.001
    
# Initialize the optimizers: SGD, Adam, RMSprop
optimizers = [tf.keras.optimizers.SGD(learning_rate), tf.keras.optimizers.Adam(learning_rate), tf.keras.optimizers.RMSprop(learning_rate)]
opt_labels = ["SGD", "Adam", "RMSprop"]
count = 0

print("Choose Optimizer: SGD(1), Adam(2), RMSprop(3)")
x = int(input())

# set optimizer
opt = optimizers[x-1]

tf.keras.backend.clear_session()

# Initialize the model.
model = MyModel()
# Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
b_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()


# Initialize lists for later visualization.
train_losses = []
test_losses = []
val_losses = []

train_accuracies = []
test_accuracies = []
val_accuracies = []


#testing once before we begin
test_loss, test_accuracy = test(model, test_dataset, b_cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check how model performs on train data once before we begin
train_loss, train_accuracy = test(model, train_dataset, b_cross_entropy_loss)
train_losses.append(train_loss)
train_accuracies.append(train_accuracy)

#check how model performs on validate data once before we begin
val_loss, val_accuracy = test(model, val_dataset, b_cross_entropy_loss)
val_losses.append(val_loss)
val_accuracies.append(val_accuracy)


# We train for num_epochs epochs
for epoch in range(num_epochs):

    #training (and checking in with training)
    epoch_loss_agg = []
    epoch_acc_agg = []
    for input,target in train_dataset:
        train_loss, train_accuracy = train_step(model, input, target, b_cross_entropy_loss, opt)
        epoch_loss_agg.append(train_loss)

        #track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))
    # tracking train accuracy
    train_accuracies.append(tf.reduce_mean(train_accuracy))

    #testing, so we can track accuracy and test loss
    test_loss, test_accuracy = test(model, test_dataset, b_cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # validation
    val_loss, val_accuracy = validate(model, val_dataset, b_cross_entropy_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f' test_losses: {test_losses[-1]} , test_accuracies: {test_accuracies[-1]}')


# Visualization
# if validate = True plotting of validate data
plotting(train_losses, test_losses, val_losses, train_accuracies, test_accuracies, val_accuracies, opt_labels[x-1], validate = False)