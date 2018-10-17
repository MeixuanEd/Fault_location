import time 
import pandas as pd
import numpy as np
from scipy import *
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv,norm
from scipy.linalg import svd, svdvals
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from io import BytesIO
from functools import partial 
from IPython.display import clear_output, Image, display, HTML  
import scipy.io as sio 
import time  
import re 

# parameters
model_name = 'Fault_location'
fault_type = 2; # 0--Tp; 1--LG; 2--LLG; 3--LL
impe_type = 1; # 1~4 fault impedance increases  
n_classes=87 
# parameters for CNN 
lambda_loss_amount = 0.001
batch_norm=1
patience = 10
learning_rate = 0.001
training_iters =9000
batch_size =50
display_step = 1000
keep_prob=0.9
dropout=0# no dropout 
decay_c =0.8
times = 1
buses =68                     
rootPath = '../01_datasets/Iu_feature'
trainName ='Line_faults_train'   
testName = 'Line_faults_test' +'_type_' + str(fault_type) + '_'+str(impe_type)
evalName = 'Line_faults_eval' 
data = sio.loadmat(os.path.join(rootPath, trainName))
linedata = data['line']   
  
def load_data(w,path,name): 
    global times, buses
    import scipy.io as sio 
    PathName = os.path.join(path, name)
    data=sio.loadmat(PathName); 
    dV= data['dV_feature'] 
    Y_ad= data['Y'] 
    train_data = (Y_ad[:,w] @ dV[w,:]).imag.T   
    train_labels = data['y_num'] 
    col, buses = np.shape(train_data)  
    train_x = np.float64(np.reshape(train_data, (int(col/times), buses,times)))  
    train_y = np.zeros((int(col/times), n_classes))
    for i in range(int(col/times)):
        train_y[i,train_labels[0][i] ] = 1;
    return train_x, train_y ,col

def choose_w(line, thres):
    global buses
    all_freq =np.zeros((1,buses))
    for i in range( buses):
        ifreq = size(np.where(line[:,0] == i+1 )) + size(np.where(line[:,1] == i+1))
        all_freq[0][i] = ifreq
    w = [i for i in range(buses) if all_freq[0][i] >thres]  
    return w

# load data
def load_all_data(w):
    global train_data, train_labels, train_num, test_data, test_labels, test_num, eval_data, eval_labels, eval_num,samples,buses, times 
    train_data, train_labels, train_num = load_data(w,rootPath, trainName)  
    eval_data, eval_labels,eval_num= load_data(w,rootPath, evalName)  
    test_data, test_labels,test_num= load_data(w,rootPath, testName)  
    samples,buses, times  = np.shape(train_data) 

# construct the classifier 
def conv2d(x, W, b, stride_row,stride_col):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, stride_row,stride_col, 1], padding='VALID') 
    x = tf.nn.bias_add(x, b)  
    return tf.nn.relu(x) # 

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv2d_norm(x,W,b, phase_train,stride_row,stride_col):
    x = tf.nn.conv2d(x, W, strides=[1, stride_row,stride_col, 1], padding='VALID')  
    x = tf.nn.bias_add(x, b)  
    x_out = batch_norm(x,1,phase_train)
    return tf.nn.relu(x_out ) 

def maxpool2d(x, height,width):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, height,  width, 1], strides=[1,  height,width, 1],
                          padding='SAME') # 
def input_weight_all(Theta,name):# Theta is a list type
    import pickle
    filepointer=open(name,"wb")
    pickle.dump(Theta,filepointer,protocol=2)
    filepointer.close()
    return 

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var) 
def conv_net(x,y, phase_train):  
    global buses, times,batch_norm, keep_prob 
    sample_num = np.shape(x)[0]
    dep1=4;dep2=8; dep3=8; dep4=8; 
    x = tf.reshape(x, shape=[-1,buses, times,1])  
    with tf.variable_scope('conv1'): 
        wc1= tf.get_variable( 'weight1',shape = [5,1,1, dep1],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc1=tf.get_variable( 'bias1',
          shape = [dep1],
          initializer=tf.constant_initializer(0.0)) 
        stride_row=1;stride_col=1
        if batch_norm:
            conv1 = conv2d_norm(x,wc1,bc1,phase_train,stride_row,stride_col)  
            conv1 = maxpool2d(conv1, 2,1)   
        else:
            conv1 = conv2d(x, wc1, bc1,stride_row,stride_col)  
            conv1 = maxpool2d(conv1, 2,1)   
        variable_summaries(wc1)
        variable_summaries(bc1)   
    with tf.variable_scope('conv2'): 
        wc2= tf.get_variable( 'weight2',shape = [5, 1, dep1, dep2],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc2=tf.get_variable( 'bias2',
          shape = [dep2],
          initializer=tf.constant_initializer(0.0)) 
        stride_row=1;stride_col=1
        if batch_norm:
            conv2 = conv2d_norm(conv1,wc2,bc2, phase_train,stride_row,stride_col ) 
            conv2 = maxpool2d(conv2, 2,1)    
        else:
            conv2 = conv2d(conv1, wc2, bc2, stride_row,stride_col) 
            conv2 = maxpool2d(conv2, 2,1)   
        variable_summaries(wc2)
        variable_summaries(bc2)   
    with tf.variable_scope('conv3'): 
        wc3= tf.get_variable( 'weight3',shape = [3, 1, dep2, dep3],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc3=tf.get_variable( 'bias3',
          shape = [dep3],
          initializer=tf.constant_initializer(0.0)) 
        stride_row=1;stride_col=1
        if batch_norm:
            conv3 = conv2d_norm(conv2,wc3,bc3, phase_train,stride_row,stride_col ) 
            conv3 = maxpool2d(conv3, 2,1)    
        else:
            conv3 = conv2d(conv2, wc3, bc3, stride_row,stride_col) 
            conv3 = maxpool2d(conv3, 2,1)   
        variable_summaries(wc3)
        variable_summaries(bc3)   
    with tf.variable_scope('conv4'): 
        wc4= tf.get_variable( 'weight4',shape = [3, 1, dep3, dep4],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc4=tf.get_variable( 'bias4',
          shape = [dep4],
          initializer=tf.constant_initializer(0.0)) 
        stride_row=1;stride_col=1
        if batch_norm:
            conv4 = conv2d_norm(conv3,wc4,bc4, phase_train,stride_row,stride_col ) 
            conv4 = maxpool2d(conv4, 2,1)    
        else:
            conv4 = conv2d(conv3, wc4, bc4, stride_row,stride_col) 
            conv4 = maxpool2d(conv4, 2,1)   
        variable_summaries(wc4)
        variable_summaries(bc4)   
        
    with tf.variable_scope('Final_out'):
        wout= tf.get_variable('wout',shape=[ 2 * dep4, n_classes],
               initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))  
        bout=tf.get_variable( 'bout',
          shape = [n_classes],
          initializer=tf.constant_initializer(0.0)) 
        # fully connected layer 
        fc1 = tf.reshape(conv4, [-1,  int(prod(conv4.get_shape()[1:])) ])   
        fc2=tf.cond( phase_train, lambda: fc1,lambda:tf.nn.dropout(fc1,keep_prob=keep_prob if dropout else 1.0)) 
        fc3= (tf.add(tf.matmul(fc2, wout), bout)) 
        
    return fc3

def establish_model(): 
    global keep_prob,learning_rate, training_iters,display_step,batch_size,patience, model_name
    global buses, times, train_data, train_labels, eval_data, eval_labels, test_data, test_labels
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, buses, times])
    y = tf.placeholder(tf.int32, [None, n_classes])  
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    
    pred = conv_net(  x, y ,phase_train) 
    y_score = tf.nn.softmax(pred)
    predict_op=tf.argmax(y_score, 1)  
    # Define loss and optimizer 
    with tf.name_scope('loss'):  
        l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() )  
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))+l2  
    with tf.name_scope('Optimizer'): 
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate ).minimize(cost) # return the pairs  of vatiables and weight
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate , decay=decay_c).minimize(cost)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    with tf.name_scope('err'): 
        correct = tf.equal(predict_op, tf.argmax(y, 1))
        err=1- tf.reduce_mean(tf.cast(correct, tf.float32))  
    tf.summary.scalar('err',err)
    tf.summary.scalar('loss',cost)
    
    #save
    saver = tf.train.Saver()
    # Launch the graph
    sess = tf.InteractiveSession()
    # Merge all the summaries and write them out to C:/Users/Lab/
    merged = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter('./train', sess.graph)
    #test_writer = tf.summary.FileWriter('./test') 
    #train_writer.add_graph(sess.graph)
    # training
    step=1
    loss_list=[]
    train_rate=[]
    eval_rate=[]
    n_incr_num =0
    best_loss = np.Inf 
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    while step < training_iters:
        ind = np.arange(train_data.shape[0])
        batch_idx = np.random.choice(ind, batch_size, replace=False) # the replace means not allow to pick up the same element again
        batch_x = train_data[batch_idx] 
        batch_y= train_labels[batch_idx] 
        indeval = np.arange(eval_data.shape[0]) 
        eval_idx = np.random.choice(indeval, batch_size, replace=False)
        batch_xeval=eval_data[eval_idx] 
        batch_yeval=eval_labels[eval_idx] 
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x,  y: batch_y,phase_train:False }) 
        loss, train_err = sess.run([cost, err], feed_dict={x: batch_x,  y: batch_y ,phase_train:False})
        loss_eval,eval_err=sess.run([cost ,err ], feed_dict={x: batch_xeval, y: batch_yeval ,phase_train:False})
        loss_list.append(loss)
        train_rate.append(train_err)
        eval_rate.append(eval_err)
        step += 1
        if step % display_step == 0:
            # Calculate batch loss and err 
            print("Iter " + str(step ) + ", Minibatch Loss= " +  "{:.2f}".format(loss) + ",training err= " + "{:.2f}".format(train_err)+ ",validating err= " + "{:.2f}".format(eval_err))   
        if loss_eval < best_loss:
                best_loss = loss_eval
                n_incr_num =0
        else:
                n_incr_num+=1
        if (n_incr_num >= patience) and (step > 7000):
            print ('Early_stopping! and the iterations is', step)
            # Create the collection.
            tf.get_collection("validation_nodes")
            # Add stuff to the collection.
            tf.add_to_collection("validation_nodes", x)  
            tf.add_to_collection("validation_nodes", phase_train) 
            tf.add_to_collection("validation_nodes", y_score)
            tf.add_to_collection("validation_nodes", y)
            results = predict_op
            #save_path = saver.save(sess, "./train/"+model_name)
            correct_results_true, results_true=sess.run([correct, predict_op] , feed_dict={x:train_data, y:train_labels,phase_train:False}) 
            total_test_err=sess.run(err , feed_dict={x:test_data,   y:test_labels,phase_train:False})  
            #print("Testing err :", total_test_err) 
            print ('Accuracy is', 100*(1-total_test_err))
            print ('Early Stop!')
            return loss_list,step,train_rate,eval_rate, correct_results_true, results_true, total_test_err
    # Create the collection.
    tf.get_collection("validation_nodes")
    # Add stuff to the collection.
    tf.add_to_collection("validation_nodes", x)   
    tf.add_to_collection("validation_nodes", phase_train) 
    tf.add_to_collection("validation_nodes", y_score)
    tf.add_to_collection("validation_nodes", y)
    #save_path = saver.save(sess, "/home/wli/03_simulations/train/"+model_name)
    correct_results_true, results_true=sess.run([correct, predict_op] ,  feed_dict={x:train_data, y:train_labels,phase_train:False}) 
    total_test_err=sess.run(err , feed_dict={x:test_data,   y:test_labels,phase_train:False})           
    #print("Testing err of subtract true :", total_test_err) 
    print ('Accuracy is', 100*(1-total_test_err))
    print("Optimization Finished!") 
    return loss_list,step,train_rate,eval_rate, correct_results_true, results_true, total_test_err  
def each_perform(correct_results,eval_labels ):
    label_y=eval_labels#np.argmax(eval_labels,1)
    acc_rate = np.zeros((1,n_classes ))
    for i in range(n_classes):
        location = np.where(label_y == i)  
        correct = [correct_results[j] for j in location[0]]  
        if len(correct) > 0:
            acc_rate[0][i] = 100*np.mean(correct)  
    return acc_rate

def plot_loss(loss,train_step,from_second,name_save, plot_name,plot_title):
    if from_second :
        plt.plot(range(0,train_step-1,1),loss[1:])
    else:
        plt.plot(range(0,train_step,1),loss[0:])
    plt.xlabel('Iterative times (t)')
    plt.ylabel(plot_name)
    plt.title(plot_title)
    plt.grid(True)
    plt.savefig(name_save)
    plt.show()
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))    

thres_degree = 0 # select those buses with larger than thres_degree degree as the inital buses   
w = choose_w(linedata, thres_degree )   
total_loss = []
total_acc = []
load_all_data(w )
loss,step,train_rate,eval_rate, correct_results_true, results_true, total_test_true  =establish_model() 
total_loss.append(loss)
total_acc.append(100*(1-total_test_true))
print (total_acc) 
show_graph(tf.get_default_graph().as_graph_def())  



