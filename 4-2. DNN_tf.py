
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
from ModelBase import *


# # Define model

# Model 1
# - No regularizer
# - 784->200->50->10
# - initializer(stddev=2)

# In[2]:

class M1(ModelBase):
    def model(self,inputs):
        with tf.variable_scope("model"):
            outputs = slim.stack(inputs,
                                 slim.fully_connected,
                                 [200,50],
                                 activation_fn=tf.sigmoid,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=2))
            outputs = slim.fully_connected(outputs,10,activation_fn=None)
        return outputs


# In[3]:

m1 = M1(modelNo=1)
m = m1
print ("Model: ",m.modelNo)
m.train()
m.plot_learning_curve()
m.plot_weight_n_bias()
m.save_out_history()


# Model 2
# - No regularizer
# - 784->400->200->100->50->25->10
# - initializer(stddev=2)

# In[4]:

class M2(ModelBase):
    def model(self,inputs):
        with tf.variable_scope("model"):
            outputs = slim.stack(inputs,
                                 slim.fully_connected,
                                 [400,200,100,50,25],
                                 activation_fn=tf.sigmoid,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=2))
        
            outputs = slim.fully_connected(outputs,10,activation_fn=None)
        return outputs


# In[5]:

m2 = M2(modelNo=2)
m = m2
print ("Model: ",m.modelNo)
m.train()
m.plot_learning_curve()
m.plot_weight_n_bias()
m.save_out_history()


# Model 3
# - Add L1-regularizer
# - 784->200->50->10
# - initializer(stddev=2)

# In[6]:

class M3(ModelBase):
    def model(self,inputs):
        with tf.variable_scope("model"):
            outputs = slim.stack(inputs,
                                 slim.fully_connected,
                                 [200,50],
                                 activation_fn=tf.sigmoid,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=2),
                                 weights_regularizer=slim.l1_regularizer(1e-8))
        
            outputs = slim.fully_connected(outputs,10,activation_fn=None)
        return outputs


# In[7]:

m3 = M3(modelNo=3)
m = m3
print ("Model: ",m.modelNo)
m.train()
m.plot_learning_curve()
m.plot_weight_n_bias()
m.save_out_history()


# Model 4
# - Add L1-regularizer
# - 784->400->200->100->50->25->10
# - initializer(stddev=2)

# In[8]:

class M4(ModelBase):
    def model(self,inputs):
        with tf.variable_scope("model"):
            outputs = slim.stack(inputs,
                                 slim.fully_connected,
                                 [400,200,100,50,25],
                                 activation_fn=tf.sigmoid,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=2),
                                 weights_regularizer=slim.l1_regularizer(1e-8))
        
            outputs = slim.fully_connected(outputs,10,activation_fn=None)
        return outputs


# In[9]:

m4 = M4(modelNo=4)
m = m4
print ("Model: ",m.modelNo)
m.train()
m.plot_learning_curve()
m.plot_weight_n_bias()
m.save_out_history()


# Model 5
# - Add L2-regularizer
# - 784->200->50->10
# - initializer(stddev=2)

# In[10]:

class M5(ModelBase):
    def model(self,inputs):
        with tf.variable_scope("model"):
            outputs = slim.stack(inputs,
                                 slim.fully_connected,
                                 [200,50],
                                 activation_fn=tf.sigmoid,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=2),
                                 weights_regularizer=slim.l2_regularizer(1e-5))
        
            outputs = slim.fully_connected(outputs,10,activation_fn=None)
        return outputs


# In[11]:

m5 = M5(modelNo=5)
m = m5
print ("Model: ",m.modelNo)
m.train()
m.plot_learning_curve()
m.plot_weight_n_bias()
m.save_out_history()


# Model 6
# - Add L2-regularizer
# - 784->400->200->100->50->25->10
# - initializer(stddev=2)

# In[12]:

class M6(ModelBase):
    def model(self,inputs):
        with tf.variable_scope("model"):
            outputs = slim.stack(inputs,
                                 slim.fully_connected,
                                 [400,200,100,50,25],
                                 activation_fn=tf.sigmoid,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=2),
                                 weights_regularizer=slim.l2_regularizer(1e-5))
        
            outputs = slim.fully_connected(outputs,10,activation_fn=None)
        return outputs


# In[13]:

m6 = M6(modelNo=6)
m = m6
print ("Model: ",m.modelNo)
m.train()
m.plot_learning_curve()
m.plot_weight_n_bias()
m.save_out_history()


# Model 7
# - Add dropout
# - 784->200->50->10
# - initializer(stddev=2)

# In[14]:

class M7(ModelBase):
    def model(self,inputs):
        with tf.variable_scope("model"):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.sigmoid):
                net = slim.fully_connected(inputs, 200)
                net = tf.nn.dropout(net,self.keep_prob)
                net = slim.fully_connected(inputs, 50)
                net = tf.nn.dropout(net,self.keep_prob)
                net = slim.fully_connected(net, 10,activation_fn=None)
                return net


# In[15]:

m7 = M7(use_droupout=True,modelNo=7)
m = m7
print ("Model: ",m.modelNo)
m.train()
m.plot_learning_curve()
m.plot_weight_n_bias()
m.save_out_history()


# Model 8
# - Add dropout
# - 784->400->200->100->50->25->10
# - initializer(stddev=2)

# In[16]:

class M8(ModelBase):
    def model(self,inputs):
        with tf.variable_scope("model"):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.sigmoid):
                net = slim.fully_connected(inputs, 400)
                net = tf.nn.dropout(net,self.keep_prob)
                net = slim.fully_connected(inputs, 200)
                net = tf.nn.dropout(net,self.keep_prob)
                net = slim.fully_connected(inputs, 100)
                net = tf.nn.dropout(net,self.keep_prob)
                net = slim.fully_connected(inputs, 50)
                net = tf.nn.dropout(net,self.keep_prob)
                net = slim.fully_connected(inputs, 25)
                net = tf.nn.dropout(net,self.keep_prob)
                net = slim.fully_connected(net, 10,activation_fn=None)
                return net


# In[17]:

m8 = M8(use_droupout=True,modelNo=8)
m = m8
print ("Model: ",m.modelNo)
m.train()
m.plot_learning_curve()
m.plot_weight_n_bias()
m.save_out_history()


# In[ ]:



