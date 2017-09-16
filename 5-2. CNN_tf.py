
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.metrics import accuracy
import time
from PreLoad_Data_n_Model import *


# In[2]:

n_epochs = 30
model_Name = 'model_3'
model_func = model_3


# ## Graph Construnction

# In[3]:

outputs, v_list = model_func(x_train_batch)
for v in v_list:
     tf.add_to_collection('vars', v) 
tf.add_to_collection('vars', outputs)

loss = tf.losses.sparse_softmax_cross_entropy(y_train_batch , outputs)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = slim.learning.create_train_op(loss, optimizer)


# In[4]:

acc = accuracy(tf.cast(tf.arg_max(outputs,1),tf.int32) , y_train_batch)


# In[5]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[6]:

_ = tf.train.start_queue_runners(sess=sess)


# ## Train

# In[7]:

ticks = time.time()

loss_history = []
acc_history = []

for epoch in range(n_epochs):
    train_loss = []
    acc_list = []
        
    for _ in range(50000/batch_size):
        _ ,loss_,acc_ = sess.run([train_op,loss,acc])
        train_loss.append(loss_)
        acc_list.append(acc_)
    
    epo_loss = np.mean(train_loss)
    epo_acc  = np.mean(acc_list)   
    loss_history.append(epo_loss)
    acc_history.append(epo_acc)
    
    if epoch % 5 == 0:
        print "epoch:%s ,train_loss:%s, acc:%s" %          (epoch ,epo_loss,epo_acc)

        
print ('Training takes '+str(time.time()-ticks)+' seconds for '+str(n_epochs)+' epochs')


# In[8]:

# Save out history
import csv
f = open('Data_history/'+model_Name+"_train_history.csv","w")
w = csv.writer(f)
w.writerows([loss_history])
w.writerows([acc_history])
f.close()


# In[9]:

# plot Learning Cruve
f = plt.figure(1, figsize=(8,3))
ax = f.add_subplot(111)
ax.set_title("Learning Curve_"+model_Name,fontsize=20)
ax.set(ylim=[0,3],ylabel='loss',xlabel='epoch')
ax.plot(range(n_epochs) , loss_history, linewidth=2.0, label='Train')
plt.savefig("Fig_byModel/Learning Curve_"+model_Name+'_train.png')
plt.show() 


# In[10]:

saver = tf.train.Saver()


# In[11]:

save_path = saver.save(sess, "Saved_Model/"+model_Name+".ckpt")


# In[ ]:



