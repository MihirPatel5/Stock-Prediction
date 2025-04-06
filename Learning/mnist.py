import numpy as np
import tensorflow as tf
from tensorflow.example.tutorials.mnist import input_data

mnist = input_data.read_datasets("MNIST_data/", one_hot=True)

input_size = 784
output_size = 10
hidden_layer = 50

tf.reset_default_graph()
 
inputs = tf.place_holder(tf.float32, [None, input_size])
target = tf.place_holder(tf.float32, [None, output_size])

weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer])
bias_1 = tf.get_variable("bias_1", [hidden_layer])
# weights_2 = tf.get_variable("weights_2")

outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + bias_1)

weights_2 = tf.get_variable("weights_2", [hidden_layer, hidden_layer])
bias_2 = tf.get_variable("bias_2", [hidden_layer])

output_2 = tf.nn.relu(tf.matmul(outputs_1, weights_2) + bias_2)

weights_3 = tf.get_variable("weights_3", [hidden_layer, output_size])
bias_3 = tf.get_variable("bias_3", [output_size])

outputs = tf.matmul(output_2, weights_3) + bias_3

loss = tf.nn.softmax_cross_entropy_with_logits(logits= outputs, lables = target)

mean_loss = tf.reduce_mean(loss)

optimize = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(mean_loss)

##Prediction Accuracy logic 

out_equal_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(target, 1))

accuracy = tf.reduce_mean(tf.cast(out_equal_target, tf.float32))

sess = tf.InteractiveSession()

initializer = tf.global_variables_initializer()

sess.run(initializer)

batch_size = 100
num_steps = 1000
batches_number  = mnist.train._num_examples // batch_size
max_epochs = 15
prev_validation_loss = 9999999.

## model learning 

for epoch_counter in range(max_epochs):
    curr_epoch_loss = 0
    for batch_counter in range(batches_number):
        input_batch, target_batch  = mnist.train.next_batch(batch_size)
        _, batch_loss = sess.run([optimize, mean_loss], feed_dict = {inputs:input_batch, target:target_batch})
        curr_batch_loss += batch_loss
    curr_batch_loss/= batches_number

    input_batch, target_batch = mnist.validation.next_batch(mnist.validation._num_examples)
    validation_loss = validation_accuracy = sess.run([mean_loss, accuracy], feed_dict ={inputs:input_batch, target:target_batch})

    print('Epoch' +str(epoch_counter+1)+
          '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss)+
          '.Validation loss:'+'{0:.3f}'.format(validation_loss)+
          '. Validation accuracy:'+'{0:.2f}'.format(validation_accuracy * 100.)+'%')
    
    if validation_loss > prev_validation_loss:
        break
    prev_validation_loss = validation_loss

print('Ending of training')

