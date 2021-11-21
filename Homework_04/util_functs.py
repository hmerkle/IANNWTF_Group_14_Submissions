# imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Description: prepare dataset-> transform dataframe to tensorflow dataset and 
#             quality to binary values and shuffle, batch and prefetch
#             Inputs: dataframe, threshold, features, target
#             Outputs: dataset
def prepare_dataset(dataframe, threshold, features, target):
    # convert dataframes into tensorflow datasets
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (tf.cast(dataframe[features].values, tf.float32), tf.cast(dataframe[target].values, tf.int32))
        )
    )
    
    # map make_binary() function to dataset
    dataset = dataset.map(lambda inputs, target: (inputs, make_binary(target, threshold)))
  
    # shuffle
    dataset = dataset.shuffle(1599)
    
    # apply batching 
    dataset = dataset.batch(32)

    # prefetch data
    dataset = dataset.prefetch(32)

    #return preprocessed dataset
    return dataset


#Description: prepare whole dataframe -> split into train, test and validate, and transform dataframe to tensorflow dataset
#             Inputs: dataframe, threshold
#             Outputs: train_dataset, test_dataset, val_dataset
def prepare_all_data(df_wine, threshold):
    # shuffle dataframe
    df_wine = df_wine.sample(frac=1).reset_index(drop=True)

    # define features and target labels
    features = list(df_wine.columns)[:-1]
    target = "quality"

    # Create a Tensorflow Dataset and a Dataset Pipeline
    ## split data frame into train(60%), test(20%), validation(20%)
    train_df, test_df, val_df = np.split(df_wine.sample(frac=1, random_state=42), [int(0.6*len(df_wine)), int(0.8*len(df_wine))])


    # convert dataframes into tensorflow datasets
    # shuffle, batch, prefetch
    train_dataset = prepare_dataset(train_df, threshold, features, target)
    test_dataset = prepare_dataset(test_df, threshold, features, target)
    val_dataset = prepare_dataset(val_df, threshold, features, target)
    
    return train_dataset, test_dataset, val_dataset



#Description: Ranks wine into good or bad
#             Inputs: target, a set threshold
#             Outputs: ranked quality of wine (binary value)
def make_binary(target, threshold):
    # wine ranked good
    if target >= tf.constant(threshold, dtype=tf.int32):
        return 1
    # wine ranked bad
    else:
        return 0
    

    
#Description: This function trains an object of the class MyModel. It conducts a forward-step and the backpropagation 
#             throughout the network. The optimizer updates weights and biases. 
#             Inputs: model, inputs, target, loss function, optimizer
#             Outputs:loss, accuracy
def train_step(model, inputs, target, loss_function, optimizer):
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(inputs)
        loss = loss_function(target, prediction)
        sample_train_accuracy =  target == np.round(prediction)
        sample_train_accuracy = np.mean(sample_train_accuracy)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, sample_train_accuracy



#Description: This function determines the test loss and test accuracy through a forward step in the network.
#             Inputs: model, test data, loss function
#             Outputs: test loss, test accuracy
def test(model, test_data, loss_function):
    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy =  target == np.round(prediction)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy



#Description: Loss and accuracy of validation dataset 
#             Inputs: model, validation data, loss function
#             Outputs: validation loss, validation accuracy
def validate(model, val_data, loss_function):
    # test over complete test data

    val_accuracy_aggregator = []
    val_loss_aggregator = []

    for (input, target) in val_data:
        prediction = model(input)
        sample_val_loss = loss_function(target, prediction)
        sample_val_accuracy =  target == np.round(prediction)
        sample_val_accuracy = np.mean(sample_val_accuracy)
        val_loss_aggregator.append(sample_val_loss.numpy())
        val_accuracy_aggregator.append(np.mean(sample_val_accuracy))

    val_loss = tf.reduce_mean(val_loss_aggregator)
    val_accuracy = tf.reduce_mean(val_accuracy_aggregator)

    return val_loss, val_accuracy



#Description: This function visualizes the losses and accuracies of training and testing 
#             Inputs: test,train,validation losses, test,train,validation accuracies, optimizer label
#             Outputs: plot
# Visualize accuracy and loss for training and test data.
def plotting(train_losses, test_losses, val_losses, train_accuracies, test_accuracies, val_accuracies, opt_label, validate):
    # plot losses
    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(test_losses)
    plt.title("Loss with optimizer " + opt_label)
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.legend((line1,line2),("Loss train","Loss test"))
    plt.show()
    
    # plot accuracies
    plt.figure()
    line1, = plt.plot(train_accuracies)
    line2, = plt.plot(test_accuracies)
    plt.title("Accuracy with optimizer " + opt_label)
    plt.xlabel("Training steps")
    plt.ylabel("Accuracy")
    plt.legend((line1,line2),("Accuracy train", "Accuracy test"))
    plt.show()
    
    if validate:
        # plot validation
        plt.figure()
        line1, = plt.plot(val_losses, "r")
        line2, = plt.plot(val_accuracies, "g")
        plt.title("Loss and Accuracy of Validation data with optimizer " + opt_label)
        plt.xlabel("Training steps")
        plt.ylabel("Loss/Accuracy")
        plt.legend((line1,line2),("Loss", "Accuracy"))
        plt.show()