import tensorflow as tf
import numpy as np
import random, math
from numpy import genfromtxt
from numpy import fft

############################### Hyperparameters #########################################
h= 60;
w= 60;
l= 5;
hidden_num= 256
class_num= 5
train_batch_size= 50
test_batch_size= 100
videoDir= './KTHdataset_0804'
labelDir= './KTHdataset_0804'
saveDir= './model_0804'

global_step= tf.Variable(tf.constant(0), trainable= False)
learningRate=tf.train.exponential_decay(learning_rate= 0.01, global_step=global_step, decay_steps= 12000, decay_rate= 0.1, staircase = True)

############################### Training data import ############################

x_all= np.zeros((4998, l, h, w))
y_all= np.zeros((4998, class_num))

x_train= np.zeros((3998, l, h, w))
y_train= np.zeros((3998, class_num))

x_test= np.zeros((1000, l, h, w))
y_test= np.zeros((1000, class_num))

for i in range(4998):
    filename= videoDir+ '/%d.csv' % (i+1)
    data= genfromtxt(filename, delimiter=',')
    for k in range(l):
        frame= data[k*h:(k+1)*h, :]
        x_all[i, k, :, :]= frame
    if i % (100) ==0:    
        print (i)

filename= labelDir+ '/labels.csv'
y_all= genfromtxt(filename, delimiter=',')

random.seed(66)
trainset_no= random.sample(range(4998), 3998)
testset_no= list(set(range(4998))-set(trainset_no))
print(trainset_no)

x_train= x_all[trainset_no]
y_train= y_all[trainset_no]

x_test= x_all[testset_no]
y_test= y_all[testset_no]

############################ Neural network model ################################
with tf.name_scope('Inputs'):
    x_= tf.placeholder(dtype=tf.float32, shape=[train_batch_size, l, h, w])
    y_= tf.placeholder(dtype=tf.float32, shape=[train_batch_size, class_num])
    x_ph_test= tf.placeholder(dtype=tf.float32, shape=[test_batch_size, l, h, w])
    y_ph_test= tf.placeholder(dtype=tf.float32, shape=[test_batch_size, class_num])

with tf.variable_scope('trainable_variables', reuse= tf.AUTO_REUSE):
    W_conv1= tf.get_variable(name= 'W_conv1', shape= [1, 3, 3, 1, 4], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
    b_conv1= tf.get_variable(name= 'b_conv1', shape= [4], initializer= tf.zeros_initializer())
    W_conv2= tf.get_variable(name= 'W_conv2', shape= [1, 3, 3, 4, 8], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
    b_conv2= tf.get_variable(name= 'b_conv2', shape= [8], initializer= tf.zeros_initializer())
    W_FC1= tf.get_variable(name= 'weight_fc1', shape= [l*hidden_num, 1024], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.001))
    b_FC1= tf.get_variable(name= 'bias_fc1', shape= [1024], initializer= tf.zeros_initializer())
    W_FC2= tf.get_variable(name= 'weight_fc2', shape= [1024, class_num], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.001))
    b_FC2= tf.get_variable(name= 'bias_fc2', shape= [class_num], initializer= tf.zeros_initializer())


with tf.name_scope('Network_model'):
    x_video= tf.reshape(x_, [train_batch_size, l, h, w, 1])
    h1= tf.nn.relu(tf.nn.conv3d(x_video, W_conv1, strides=[1,1,1,1,1], padding='SAME')+ b_conv1)
    h1_pool= tf.nn.max_pool3d(h1, ksize= [1, 1, 2, 2, 1], strides= [1, 1, 2, 2, 1], padding= 'SAME')
    h2= tf.nn.relu(tf.nn.conv3d(h1_pool, W_conv2, strides=[1,1,1,1,1], padding='SAME')+ b_conv2)
    h2_pool= tf.nn.max_pool3d(h2, ksize= [1, 1, 2, 2, 1], strides= [1, 1, 2, 2, 1], padding= 'SAME')
    h3_flatten= tf.reshape(h2_pool, [train_batch_size, l, 15*15*8])
    cell= tf.nn.rnn_cell.BasicLSTMCell(hidden_num)
    zero_state= cell.zero_state(batch_size= train_batch_size, dtype= tf.float32)
    h4, final_state= tf.nn.dynamic_rnn(cell= cell, initial_state= zero_state, inputs= h3_flatten, dtype= tf.float32, scope= 'rnn_1')
    h4_reshape= tf.reshape(h4, [train_batch_size, l*hidden_num])
    h5= tf.nn.relu(tf.matmul(h4_reshape, W_FC1)+ b_FC1)
    output= tf.nn.softmax(tf.matmul(h5, W_FC2)+ b_FC2)
    train_loss= -tf.reduce_sum(y_* tf.log(output)) + 0.005*tf.reduce_sum(tf.square(W_conv2))
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
    accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, 'float')) 

with tf.name_scope('Network_test'):
    x_video_test= tf.reshape(x_ph_test, [test_batch_size, l, h, w, 1])
    h1_test= tf.nn.relu(tf.nn.conv3d(x_video_test, W_conv1, strides=[1,1,1,1,1], padding='SAME')+ b_conv1)
    h1_pool_test= tf.nn.max_pool3d(h1_test, ksize= [1, 1, 2, 2, 1], strides= [1, 1, 2, 2, 1], padding= 'SAME')
    h2_test= tf.nn.relu(tf.nn.conv3d(h1_pool_test, W_conv2, strides=[1,1,1,1,1], padding='SAME')+ b_conv2)
    h2_pool_test= tf.nn.max_pool3d(h2_test, ksize= [1, 1, 2, 2, 1], strides= [1, 1, 2, 2, 1], padding= 'SAME')
    h3_flatten_test= tf.reshape(h2_pool_test, [test_batch_size, l, 15*15*8])
    zero_state_test= cell.zero_state(batch_size= test_batch_size, dtype= tf.float32)
    h4_test, final_state_test= tf.nn.dynamic_rnn(cell= cell, initial_state= zero_state_test, inputs= h3_flatten_test, dtype= tf.float32, scope= 'rnn_1')
    h4_reshape_test= tf.reshape(h4_test, [test_batch_size, l*hidden_num])
    h5_test= tf.nn.relu(tf.matmul(h4_reshape_test, W_FC1)+ b_FC1)
    output_test= tf.nn.softmax(tf.matmul(h5_test, W_FC2)+ b_FC2)
    test_loss= -tf.reduce_sum(y_ph_test* tf.log(output_test)) + 0.005*tf.reduce_sum(tf.square(W_conv2))
    correct_prediction_test = tf.equal(tf.argmax(output_test, 1), tf.argmax(y_ph_test, 1))
    accuracy_test= tf.reduce_mean(tf.cast(correct_prediction_test, 'float'))

tf.summary.scalar('Training_Loss', train_loss)
tf.summary.scalar('Valid_Loss', test_loss)
train_step= tf.train.GradientDescentOptimizer(learningRate).minimize(train_loss, global_step= global_step)

sess= tf.Session()
saver=tf.train.Saver()
writer=tf.summary.FileWriter('./summary', sess.graph)
merged=tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

with open(saveDir+ '/Loss.txt', 'w') as f_loss:
	f_loss.write('')

with open(saveDir+ '/Loss.txt', 'a') as f_loss:
	total_epoch= 10000
	for i in range(total_epoch):
		mini_1= random.sample(range(3998), train_batch_size)
		x_feed1= x_train[mini_1]
		y_feed1= y_train[mini_1]
		mini_test= random.sample(range(1000), test_batch_size)
		x_test_feed= x_test[mini_test]
		y_test_feed= y_test[mini_test]
		if i % (total_epoch/100) ==0:
			trainResult=sess.run(merged, feed_dict={x_: x_feed1, y_: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed})
			writer.add_summary(trainResult, i)
			f_loss.write(str(sess.run(train_loss, feed_dict={x_: x_feed1, y_: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed})))
			f_loss.write(',')
			f_loss.write(str(sess.run(test_loss, feed_dict={x_: x_feed1, y_: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed})))
			f_loss.write('\n')
			print('Training Process:', i/total_epoch*100, '%')
			print('learningrate:', sess.run(learningRate))
			print('trainLoss:', sess.run(train_loss, feed_dict={x_: x_feed1, y_: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed}))
			print('train_acc:', sess.run(accuracy_train, feed_dict={x_: x_feed1, y_: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed}))
			print('testLoss:', sess.run(test_loss, feed_dict={x_: x_feed1, y_: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed}))
			print('test_acc:', sess.run(accuracy_test, feed_dict={x_: x_feed1, y_: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed}))
		sess.run(train_step, feed_dict={x_: x_feed1, y_: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed})

print('Training Finished')

savePath=saver.save(sess, saveDir+ '/parameters.ckpt')
print('Model params saved in: ', savePath)








