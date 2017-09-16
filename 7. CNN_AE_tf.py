
# coding: utf-8

# In[1]:

model_name = 'Copy7' 


# In[2]:

get_ipython().magic(u'matplotlib inline')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import random
import os
import sys
import pickle


# ### Read Data

# In[3]:

def load_data():
    MNIST_M = np.load('MNIST_M.npy')
    train_data, train_label = MNIST_M[0]
    valid_data, valid_label = MNIST_M[1]
    test_data, test_label = MNIST_M[2]
    
    return train_data, train_label, valid_data,valid_label, test_data,test_label

# In[4]:

train_data, train_label, valid_data, valid_label, test_data, test_label = load_data()


# In[5]:

train_data.shape


# ### Define Model

# add pooling , and discrib why use it

# In[6]:

def max_unpool_2x2(x , size ):
    inference = tf.image.resize_nearest_neighbor(x, size )
    return inference

'''
x = tf.reshape(tf.constant(list(range(100*3*3*256))), [100,3,3,256])
y = max_unpool_2x2(x)
z = tf.shape(y)
k =tf.Print(z,[z])
sess=tf.InteractiveSession()
sess.run(k)
'''


# In[7]:

def lrelu(x, leaky=0.2): # leaky relu
    return tf.maximum(x, leaky * x)


# In[8]:

def encoder(x): # input:  (batch, height, width, channels)= (?, 28, 28, 3) 
    with tf.variable_scope('encoder'):
        with slim.arg_scope([slim.conv2d],
                            activation_fn = tf.nn.sigmoid,
                            kernel_size=[4,4],
                            padding='SAME',
                            stride=1):
            #layer 1
            enc1 = slim.conv2d(x , 64, scope='enc1') #(?, 28, 28, 3) --> (?, 28, 28, 64)
            print('enc1', enc1.shape)
            poo1 = slim.max_pool2d(enc1, [2,2], scope='poo1')  #(?, 28, 28, 64) --> (?, 14, 14, 64) 
            print('poo1',poo1.shape)
            
            #layer 2
            enc2 = slim.batch_norm(slim.conv2d(poo1 , 128, scope='enc2'))# (?, 14, 14, 64)-->(?, 14, 14, 128)
            print('enc2',enc2.shape)
            poo2 = slim.max_pool2d(enc2, [2,2], scope='poo2')  #(?, 14, 14, 128) --> (?, 7, 7, 128) 
            print('poo2',poo2.shape)       
            
            #layer 3
            enc3 = slim.batch_norm(slim.conv2d(poo2 , 256, scope='enc3'))# (?, 7, 7, 128)--> (?, 7, 7, 256) 
            print('enc3',enc3.shape)
            poo3 = slim.max_pool2d(enc3, [2,2], scope='poo3')  #(?, 7, 7, 256) --> (?, 3, 3, 256) 
            print('poo3',poo3.shape)    
            
            #layer 4
            enc4 = slim.batch_norm(slim.conv2d(poo3 , 512, scope='enc4'))#(?, 3, 3, 256) --> (?, 3, 3, 512) 
            print('enc4',enc4.shape)
            
            return enc4


# In[9]:

def decoder(enc):
    with tf.variable_scope('decoder'):
        with slim.arg_scope([slim.conv2d_transpose],
                            activation_fn = tf.nn.sigmoid,
                            kernel_size=[4,4],
                            padding='SAME',
                            stride=1):
            
            #layer 1
           
            dec1 = slim.batch_norm(slim.conv2d_transpose(enc ,256, scope='dec1'))#(?, 3, 3, 256) 
            print('dec1',dec1.shape)
            unp1 = max_unpool_2x2(dec1, [6,6])#(?, 6, 6, 256) 
            print('unp1',unp1.shape)  
 
            
            #layer 2
            dec2 = slim.batch_norm(slim.conv2d_transpose(unp1 ,128, scope='dec2'))#(?, 6, 6, 128) 
            print('dec2',dec2.shape)
            unp2 = max_unpool_2x2(dec2, [12,12])  #(?, 12, 12, 128) 
            print('unp2',unp2.shape) 
            
            #layer 3
            dec3 = slim.batch_norm(slim.conv2d_transpose(unp2 ,64, scope='dec3')) #(?, 12, 12, 64) 
            print('dec3',dec3.shape)
            unp3 = max_unpool_2x2(dec3, [24,24])   #(?, 24, 24, 64) 
            print('unp3',unp3.shape)

            #layer 4
            dec4 = slim.batch_norm(slim.conv2d_transpose(unp3 ,32, scope='dec4')) #(?, 24, 24, 32)  
            print('dec4',dec4.shape)

            #layer 5 : resize
            dec5 = slim.conv2d_transpose(dec4 , 3, scope='dec5',
                                        kernel_size=[5,5],
                                        padding='VALID',
                                        activation_fn=tf.nn.tanh)
                                            #(?, 24, 24, 32) --> (?, 28, 28, 3)
                                            # 24x24 -> 28x28  using padding='VALID' and kernal_size=[5,5]
                                            # tanh: contrain range between [-1,1]
            print('dec5',dec5.shape)
            
            return dec5


# ### Construct Graph

# In[10]:

data = tf.placeholder(tf.float32,[None ,28 ,28 ,3 ])
norm_data = tf.cast(data,tf.float32) / 127.5 -1  # int8--> float32, and contrain range between [-1,1]


# In[11]:

enc = encoder(norm_data)
rec = decoder(enc)


# In[12]:

loss = tf.reduce_mean((norm_data - rec)**2) # L-2 loss
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)


# In[13]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# ### Training & plot loss history

# In[14]:

def save_model(save_sess, save_path):
    print('>> Save model...')
    saver = tf.train.Saver()
    saver.save(save_sess, save_path ) 
#path = 'ckpt/unpoolModel_epo_%s.ckpt' % 0
#save_model(sess, path)


# In[15]:

def sample_data (data, label):
    # 0~9 indexes
    idxs_list = []
    for i in range(10):
        idxs = np.where(label[:,i] == 1 )[0]  
        idxs_list.append( idxs )
        #print (idxs_list[i].shape)
        
    # 0~9 sample image
    sample_data_ = []
    for i in range(10):
        idx = random.sample(idxs_list[i],1)
        dt = data [idx]
        sample_data_.append(dt)
    sample_data_ = np.squeeze(sample_data_)   
    return idxs_list, sample_data_

def plot_10_images(data ,msg=''):
    print(msg)
    fig = plt.figure(figsize=(5,2) )    
    for i in range(10):
        a=fig.add_subplot(2,5,i+1 )
        plt.imshow(data[i])
        a.set_axis_off()

    plt.show()
    plt.close() 
    


# In[16]:

idxs_list, sample_data_ = sample_data (test_data, test_label)
plot_10_images(sample_data_ ,msg='Original sample image:')


# In[17]:

batch_size = 100
epochs = 30
loss_hist_train = []
loss_hist_valid = []
save_nb_epos = 3
for epoch in range(epochs):
    
    # train
    loss_ = []
    for batch in range(train_data.shape[0] / batch_size):
        batch_data = train_data[batch:batch+batch_size]
        
        _,loss_i = sess.run([train_op , loss] ,feed_dict={data:batch_data})
        loss_.append(loss_i)
    
    loss_hist_train.append( np.mean(loss_) )
    
    # valid
    loss_valid = sess.run(loss ,feed_dict={data:valid_data})
    loss_hist_valid.append( loss_valid )

    # 後處理: 每3回合, 或最後一回合
    if  (epoch % save_nb_epos ==0) or (epoch == epochs-1):

        # save model
        path = 'ckpt/%s_unpoolModel_epo_%s.ckpt' % (model_name, epoch) 
        save_model(sess, path)
        
        # Reconstructed Images (from Gaussian) during training
        rec_ = sess.run(rec , feed_dict={data: sample_data_ })
        rec_img = [ np.clip((r+1)*127.5 , 0,255).astype('uint8')for r in rec_ ]
        plot_10_images(rec_img ,msg='Reconstructed sample image:')

 
    print ('Epoch: %s , train_loss: %s , valid_loss: %s ' % (epoch, loss_hist_train[-1], loss_hist_valid[-1] ) )    


# ### (a) Plot the reconstruction loss

# In[18]:

#epochs = 10
#loss_hist_train = range(epochs)
#loss_hist_valid = range(epochs)

# plot
f = plt.figure(1)
ax = f.add_subplot(111)
ax.plot(range(epochs),loss_hist_train,label='train loss')
ax.plot(range(epochs),loss_hist_valid,label='valid loss')
ax.set(xlabel='epoch',ylabel='loss')
ax.legend(loc='center right')
f.savefig('AE_loss_hist.png')


# In[19]:

# save loss
import csv
with open('AE_loss_hist.csv', 'wb') as csvfile:
    w = csv.writer(csvfile)
    w.writerow(['Epoch'] + range(epochs))
    w.writerow(['loss_hist_train'] + loss_hist_train)
    w.writerow(['loss_hist_valid'] + loss_hist_valid)


# ### (b) Plot 10 random samples of reconstruction image of test data and original image together

# In[ ]:

idxs_list, sample_data_ = sample_data (test_data, test_label)
plot_10_images(sample_data_ ,msg='Original sample image:')


# In[ ]:

# load sess
sess = tf.Session() 
saver = tf.train.import_meta_graph('./ckpt/%s_unpoolModel_epo_%s.ckpt.meta' % (model_name, 3) )
saver.restore(sess,tf.train.latest_checkpoint('./ckpt/'))


# In[ ]:

# Reconstructed Images (from Gaussian) during training
rec_ = sess.run(rec , feed_dict={data: sample_data_ })
rec_img = [ np.clip((r)*255 , 0,255).astype('uint8')for r in rec_ ]
plot_10_images(rec_img ,msg='Reconstructed sample image:')


# ### Back up

# In[ ]:

rec_ = sess.run(rec , feed_dict={data:test_data})


# In[ ]:

rec_img = np.clip((rec_[2]+1)*127.5 , 0,255).astype('uint8') 


# In[ ]:

plt.imshow(rec_img)


# In[ ]:

plt.imshow(test_data[2])


# In[ ]:



