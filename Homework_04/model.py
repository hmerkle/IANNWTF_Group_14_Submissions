import tensorflow as tf

# Description: The class MyModel describes a multi-layer perceptron with two hidden layers and one output layer. Regularizers and dropout 
#              included
class MyModel(tf.keras.Model):
    
    # Description: Set up layers and define regularizers and dropout
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(11, activation=tf.nn.relu, kernel_regularizer=  tf.keras.regularizers.l2(0.001))
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.hidden_layer2 = tf.keras.layers.Dense(11, activation=tf.nn.relu, kernel_regularizer=  tf.keras.regularizers.l2(0.001))
        self.output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    # Description: First a python decorater is called to transform the call function to a computational graph
    #              Then the call function passes the inputs through the layers of the model (forward pass)
    #              @parameters: input, training label
    #              @returns: final_pass
    @tf.function
    def call(self, inputs, training = True):
        x = self.hidden_layer1(inputs)
        x = self.hidden_layer2(x)
        if training:
            x = self.dropout(x, training=training)    
        x = self.output_layer(x)
        return x