
# coding: utf-8

# hidden dim = 100
# 

# In[1]:

model_name = 'dim100_epoch500'
fname = 'files_'+model_name+'/'


# In[2]:

get_ipython().magic(u'matplotlib inline')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import random
from sklearn.manifold import TSNE
import matplotlib.cm as cm


# ### Define Model 

# In[3]:

class AAE():
    def __init__(self):
        # read data
        '''
        mnist.train.images 是一個數量為55000, mean = 0.1307, sigma = 0.3082的datasets
        值大部分很接近0(黑色), 少部分很接近1(白色), 介於中間的很少 
        '''
        print('\n>> Read data...')
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        self.N = len(self.mnist.train.images) #data數目
        # build_graph
        self.run_epoch = 0
        print('\n>> Build graph...')
        self.build_graph()
        # log
        self.rec_loss_history = []
        self.d_loss_history = []
        self.enc_loss_history = []
        self.latent_mean = []
        self.latent_std = []
        print('\n>> Build graph ok!')
        
    def encoder(self,x):
        with tf.variable_scope('encoder'):
            x = slim.fully_connected(x, 400)
            x = slim.fully_connected(x, 100 ,activation_fn=None)
            return x

    def decoder(self,x):
        with tf.variable_scope('decoder'):
            x = slim.fully_connected(x, 200)
            x = slim.fully_connected(x, 784, activation_fn = tf.nn.sigmoid)

            return x

    def discriminator(self,x,reuse=False):
        with tf.variable_scope('discriminator',reuse=reuse):
            x = slim.fully_connected(x, 200)
            x = slim.fully_connected(x, 100)
            logits = slim.fully_connected(x, 1, activation_fn=None)

            return tf.nn.sigmoid(logits) , logits

    def build_graph(self):
        # placeholder
        self.data = tf.placeholder(tf.float32,[None ,28*28])
        self.z = tf.placeholder(tf.float32,[None ,100])
        # graph: AE
        self.enc = self.encoder(self.data)
        self.rec = self.decoder(self.enc)
        # rec_loss: AE(Autoencoder)把圖片重建得越像越好
        self.rec_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.data - self.rec),1))
        # graph: D
        p_real,real_logits = self.discriminator(self.z)# real Gauss 
        p_fake,fake_logits = self.discriminator(self.enc,reuse=True)# fake Gauss (ENC產生的)
        # d_loss: D(discriminator)判斷real normal或fake normal(ENC產生的normal), 判斷得越準越好
        d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(p_real),#real標1
                                                                logits=real_logits))
        d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(p_fake),#fake標0
                                                                logits=fake_logits))
        self.d_loss = d_loss_real + d_loss_fake
        # enc_loss: ENC(encoder)產生的latent分布越接近real normal越好
        self.enc_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(p_fake),
                                                                logits=fake_logits))
                                                                #假設D是準的, ENC的output要盡量讓D判real
        # optimizer 
        optimizer = tf.train.AdamOptimizer(0.0001)
        # param
        AE_params = slim.get_variables(scope='encoder')+slim.get_variables(scope='decoder')
        d_params = slim.get_variables(scope='discriminator')
        enc_params = slim.get_variables(scope='encoder')
        # trainer
        '''
            # AE(Autoencoder)把圖片重建得越像越好
            # ENC(encoder)產生的latent分布越接近real normal越好
            # D(discriminator)判斷real normal或fake normal(ENC產生的normal), 判斷得越準越好
        '''
        self.AE_trainer = optimizer.minimize(self.rec_loss, var_list=AE_params)
        self.D_trainer = optimizer.minimize(self.d_loss, var_list=d_params)
        self.ENC_trainer = optimizer.minimize(self.enc_loss, var_list=enc_params) 
        # run sess
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  
    
    def train(self, epochs = 50, batch_size = 128, save_nb_epos =5 ):
        print('\n>> Start training...')
        p_z = np.random.normal(3,1,size=(10,100)) #每回合產出10張圖, latent space 100維
        for epoch in range(epochs):

            rec_loss_l = []
            d_loss_l = []
            enc_loss_l = []

            for batchNo in range(self.N/batch_size):
                batch_xs, batch_ys = self.mnist.train.next_batch(batch_size) #取一個新batch
                # AE(Autoencoder)把圖片重建得越像越好
                # D(discriminator)判斷real normal或fake normal(ENC產生的normal), 判斷得越準越好
                batch_z = np.random.normal( 3, 1,size=(batch_size,100)) # mu, sigma調整, 使sample的值>0
                
                _,rec_loss_,_,d_loss_  = self.sess.run(   [self.AE_trainer,self.rec_loss,
                                                           self.D_trainer,self.d_loss ],
                                                           feed_dict={self.data: batch_xs,
                                                                      self.z: batch_z })
                # ENC(encoder)產生的latent分布越接近real normal越好
                _,enc_loss_,enc_ = self.sess.run( [self.ENC_trainer,  self.enc_loss,  self.enc],
                                                  feed_dict={self.data: batch_xs,
                                                             self.z: batch_z })

                rec_loss_l.append(rec_loss_)
                d_loss_l.append(d_loss_)
                enc_loss_l.append(enc_loss_)
            
            # loss history
            m_rec_loss = np.mean(rec_loss_l)
            m_d_loss = np.mean(d_loss_l)
            m_enc_loss = np.mean(enc_loss_l)

            self.rec_loss_history.append(m_rec_loss)
            self.d_loss_history.append(m_d_loss)
            self.enc_loss_history.append(m_enc_loss)
            
            latent_mean =np.mean(enc_)
            latent_std  =np.std(enc_) 
            self.latent_mean.append(latent_mean) 
            self.latent_std.append(latent_std)

            print ">> epoch:%s ,rec_loss:%s ,d_loss:%s ,enc_loss:%s" % (epoch , m_rec_loss, m_d_loss, m_enc_loss)

            # 後處理: 每50回合, 或最後一回合
            if  (epoch % save_nb_epos ==0) or (epoch == epochs-1):
                # show latent space info
                print ('           latent_mean: %s,latent_std: %s,' %( latent_mean, latent_std) ) 
                
                # save model
                path = fname +'ckpt/%s_runEpo_%s.ckpt' % (model_name, epoch+1) 
                self.save_model(path)

                # reconstruct picture during training    
                self.reconstruct_picture(p_z, epoch=epoch)
                
                # reduced_dim 100 to 2 and plot:看分群狀況     
                #self.reduced_dim_100to2_plot(batch_xs, batch_ys,epoch=epoch)
                
            self.run_epoch+=1
                
        print('\n>> ...End training, %s epochs so far' % self.run_epoch)
    
    def save_model(self, save_path):
        print('\n>> Save model...')
        saver = tf.train.Saver()
        saver.save(self.sess, save_path ) 
        
    # load model
    def restore_last_session(self, save_path=fname +'ckpt/'):
        print('\n>> restore_last_session...')
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(save_path)
        print(ckpt.model_checkpoint_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess
        
    def reconstruct_picture(self, p_z, epoch=-1 ):
        print('\n>> Reconstruct_picture...')
        rec_ = self.sess.run( self.rec ,feed_dict={self.enc: p_z })
        for i, r in enumerate(rec_):
            save_path = fname +'rec_pic/picNo_%s_epo_%s.png' % (i,epoch)
            plt.imsave(save_path, r.reshape(28,28),cmap='gray')  
            
    def reduced_dim_100to2_plot(self, data_, label_, epoch=-1 ):
        # Plot the output of encoder of training data, Reduce dim from 100 to 2
        print('\n>> Reduced_dim_ 100 to 2 and plot...')
        '''
        z is the output of encoder
        z = model.encoding(sess, data)
        '''
        z = self.sess.run(self.enc, feed_dict={self.data:data_})
        tsne = TSNE(n_components = 2, random_state = 0)
        t_z = tsne.fit_transform(z)
        '''
        plot the t_z, color is determined by label
        '''
        colors = cm.rainbow(np.linspace(0, 1, 10))
        scatter = []
        index=range(10)
        f = plt.figure(1)
        ax = f.add_subplot(111)
        for i in index:
            tmp = np.where(label_ == i) 
            scatter.append(ax.scatter(t_z[tmp, 0], t_z[tmp, 1], c = colors[i] ,s = 5. ,linewidths=0))
        ax.legend(scatter, index)
        f.savefig(fname +'reduced_dim/reduced_dim_epo_%s.png'%epoch)
        plt.show()        
        


# ### Read data & Build up Graph

# In[4]:

aae = AAE()


# ### Training

# In[5]:

aae.train(epochs = 500, save_nb_epos = 50)


# In[6]:

aae.run_epoch


# ### (a) Plot the output of encoder of training data, Reduce dim from 100 to 2

# In[7]:

sp_idx = random.sample(range(aae.N),2000) #sample 2000 data
data_ = aae.mnist.train.images[sp_idx] 
label_ = aae.mnist.train.labels[sp_idx]
aae.reduced_dim_100to2_plot(data_, label_,epoch=aae.run_epoch)


# ### (b) Plot loss history

# In[9]:

rec_loss_history = aae.rec_loss_history
d_loss_history =aae.d_loss_history
enc_loss_history =aae.enc_loss_history

x_ =  range(len(aae.rec_loss_history))

# plot
f = plt.figure(1)
ax = f.add_subplot(111)
ax.set_title('Reconstruction loss')
ax.plot(x_,aae.rec_loss_history,label='reconstruction')
ax.set(xlabel='epoch',ylabel='loss')
ax.legend(loc='best')
f.savefig(fname +'AAE_rec_loss_hist.png')

# plot
f = plt.figure(2)
ax = f.add_subplot(111)
ax.set_title('Discriminator loss')
ax.plot(x_,aae.d_loss_history,label='discriminator')
ax.set(xlabel='epoch',ylabel='loss')
ax.legend(loc='best')
f.savefig(fname +'AAE_d_loss_hist.png')

# plot
f = plt.figure(3)
ax = f.add_subplot(111)
ax.set_title('Generator loss')
ax.plot(x_,aae.enc_loss_history,label='encoder')
ax.set(xlabel='epoch',ylabel='loss')
ax.legend(loc='best')
f.savefig(fname +'AAE_enc_loss_hist.png')

# plot
f = plt.figure(4)
ax = f.add_subplot(111)
ax.set_title('Latent Mean and std')
ax.plot(x_,aae.latent_mean,label='latent_mean')
ax.plot(x_,aae.latent_std,label='latent_std')
ax.set(xlabel='epoch',ylabel='')
ax.legend(loc='best')
f.savefig(fname +'AAE_Latent_Mean_and_std.png')


# save loss
import csv
with open(fname +'AAE_loss_hist.csv', 'wb') as csvfile:
    w = csv.writer(csvfile)
    w.writerow(['Epoch'] + x_)
    w.writerow(['Reconstruction loss'] + aae.rec_loss_history)
    w.writerow(['Discriminator loss'] + aae.d_loss_history)
    w.writerow(['Generator loss'] + aae.enc_loss_history)
    w.writerow(['latent_mean'] + aae.latent_mean)
    w.writerow(['latent_std'] + aae.latent_std)


# ### (c) Show Reconstructed Images (Sampling)

# In[102]:

import matplotlib.image as mpimg

# Reconstructed Images (from Gaussian) during training
for epo in list(range(0,500,50))+[499]:
    print('>> epoch %s:'%epo)
    fig = plt.figure(figsize=(5,2) )
    for i in range(10):
        save_path = fname +'rec_pic/picNo_%s_epo_%s.png' % (i,epo)

        a=fig.add_subplot(2,5,i+1, )
        img = mpimg.imread(save_path)
        plt.imshow(img,cmap='gray')
        a.set_axis_off()
    plt.show()
    plt.close()



# In[117]:

# 0~9 indexes
idxs_list = []
for i in range(10):
    idxs = np.where(label_ == i )[0]  
    idxs_list.append( idxs )


# In[118]:

# 0~9 average image
mean_data_ = []
for i in range(10):
    dt = np.mean(data_[idxs_list[i]],axis=0)
    mean_data_.append(dt)
    
print('mean pic:')
fig = plt.figure(figsize=(5,2) )    
for i in range(10):
    a=fig.add_subplot(2,5,i+1 )
    plt.imshow(mean_data_[i].reshape(28,28),cmap='gray')
    a.set_axis_off()
    
plt.show()
plt.close()


# In[119]:

#tiny = np.random.normal(0,0.1,size=(10,28,28))
#z =  np.random.normal( 3, 1,size=(10,100))
#enc_ = aae.sess.run(aae.enc, feed_dict={aae.data:mean_data_})
#rec_ = aae.sess.run(aae.rec, feed_dict={aae.enc:enc_,aae.z:z })
rec_ = aae.sess.run(aae.rec, feed_dict={aae.data:mean_data_ })


# In[121]:

print('reconstructed mean pic:')

fig = plt.figure(figsize=(5,2) )    
for i in range(10):
    a=fig.add_subplot(2,5,i+1 )
    plt.imshow(rec_[i].reshape(28,28),cmap='gray')
    a.set_axis_off()
    
plt.show()
plt.close()


# In[125]:

# 0~9 sample image
sample_data_ = []
for i in range(10):
    idx = random.sample(idxs_list[i],1)
    dt = data_[idx]
    sample_data_.append(dt)
sample_data_ = np.squeeze(sample_data_)   

print('sample pic:')
fig = plt.figure(figsize=(5,2) )    
for i in range(10):
    a=fig.add_subplot(2,5,i+1 )
    plt.imshow(sample_data_[i].reshape(28,28),cmap='gray')
    a.set_axis_off()
    
plt.show()
plt.close()


# In[126]:

rec_ = aae.sess.run(aae.rec, feed_dict={aae.data:sample_data_ })


# In[127]:

print('reconstructed sample pic:')

fig = plt.figure(figsize=(5,2) )    
for i in range(10):
    a=fig.add_subplot(2,5,i+1 )
    plt.imshow(rec_[i].reshape(28,28),cmap='gray')
    a.set_axis_off()
    
plt.show()
plt.close()


# conclusion: 
# - AE are not good enough, so numbers like 0,4,5,6 can't be reconstructed well, but rec_loss can't be lower anymore

# ### Back up

# In[ ]:

#aae.sess = aae.restore_last_session(fname+'ckpt/')


# In[ ]:



