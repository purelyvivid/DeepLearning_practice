import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import csv

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

class ModelBase:
    def __init__(self,use_droupout=False,modelNo=0):
        self.modelNo = str(modelNo)
        self.images = tf.placeholder(tf.float32, [None, 28*28])
        self.labels = tf.placeholder(tf.int32, [None])
        self.use_droupout = False
        if use_droupout:
            self.use_droupout = True
            self.keep_prob = tf.placeholder(tf.float32)
        
        self.outputs = self.model(self.images)
        
        self.params = slim.get_variables(scope='model')
        loss = tf.losses.sparse_softmax_cross_entropy(self.labels,self.outputs)
        self.total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = slim.learning.create_train_op(self.total_loss, optimizer)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.training_history = []
        self.test_history = []
        self.epochs = 100
        self.params_ = self.sess.run(self.params)
        
    def model(self,inputs):
        pass
        
    
    def train(self,keep_prob=0.8):
        batch_size = 32
        train_step_per_epoch = mnist.train.num_examples / batch_size
        test_images , test_labels = mnist.test.next_batch(10000)
        
        
        for epoch in range(self.epochs):
            total_train_costs = []
            
            for idx in xrange(train_step_per_epoch):
                batch_images , batch_labels = mnist.train.next_batch(batch_size)
                feed_dict={self.images: batch_images,
                           self.labels: batch_labels}
                if self.use_droupout:
                    feed_dict[self.keep_prob]=keep_prob
                    
                _ , loss_ = self.sess.run([self.train_op , self.total_loss],
                                          feed_dict=feed_dict)
                total_train_costs.append(loss_)
    
            
            feed_dict={self.images: test_images,
                       self.labels: test_labels}
        
            if self.use_droupout:
                feed_dict[self.keep_prob]=keep_prob
                
            test_loss = self.sess.run(self.total_loss,
                                      feed_dict=feed_dict)
    
            if epoch % 10 == 0:
                print "epoch:%s ,train_loss:%s ,test_loss:%s " % (epoch ,np.mean(total_train_costs) , test_loss)
    
            self.training_history.append(np.mean(total_train_costs))
            self.test_history.append(test_loss)
        
        self.params_ = self.sess.run(self.params)
        tf.reset_default_graph()
        
    def plot_learning_curve(self):
        f = plt.figure(1, figsize=(8,3))
        ax = f.add_subplot(111)
        ax.set_title("Learning Curve_model"+self.modelNo,fontsize=20)
        ax.set(ylim=[0,1],ylabel='loss',xlabel='epoch')
        ax.plot(range(self.epochs) , self.training_history, linewidth=2.0, label='Train')
        ax.plot(range(self.epochs) , self.test_history, linewidth=2.0, label='Test')
        ax.legend(loc='best', frameon=True,fontsize=8) #show label
        plt.savefig("Fig_byModel/Learning Curve_model"+self.modelNo+'.png')
        plt.show() 
    
    def plot_weight_n_bias(self):
        # plot all in one
        f, sub_plt = plt.subplots(2, 1, figsize=(8,10))
        is_weight = True
        paraNo = 0
        for p in self.params_:
            layerNo = str(paraNo/2+1)
            if is_weight:
                _ ,_,_=sub_plt[0].hist(p.flatten(),100,label='layer_'+layerNo,range=[-5,5])
            else:
                _ ,_,_=sub_plt[1].hist(p.flatten(),100,label='layer_'+layerNo,range=[-5,5])
            is_weight = ( is_weight == False )
            paraNo+=1
         
        sub_plt[0].set_title("model"+self.modelNo,fontsize=20)
        sub_plt[1].set_title("model"+self.modelNo,fontsize=20)
        sub_plt[0].set(ylim=[0,7000],ylabel='weight')
        sub_plt[1].set(ylim=[0,50],ylabel='bias')
        sub_plt[0].legend(loc='best', frameon=True,fontsize=8) #show label
        sub_plt[1].legend(loc='best', frameon=True,fontsize=8) #show label
        plt.savefig("Fig_byModel/weight_n_bias_model"+self.modelNo+'.png')
        plt.show()  
        plt.close()
        
        # plot one
        is_weight = True
        paraNo = 0
        for p in self.params_:
            if is_weight:
                f, sub_plt = plt.subplots(2, 1, figsize=(8,10))
                layerNo = str(paraNo/2+1)
                _ ,_,_=sub_plt[0].hist(p.flatten(),100,label='layer_'+layerNo,range=[-5,5])
                sub_plt[0].set_title("Weight_model"+self.modelNo+'_layer'+layerNo,fontsize=20)
                sub_plt[0].set(ylabel='weight')
            else:
                _ ,_,_=sub_plt[1].hist(p.flatten(),100,label='layer_'+layerNo,range=[-5,5])
                sub_plt[1].set_title("Bias_model"+self.modelNo+'_layer'+layerNo,fontsize=20)
                sub_plt[1].set(ylabel='bias') 
                plt.savefig("Fig_byLayer/weight_n_bias_model"+self.modelNo+'_layer'+layerNo+'.png')
                #plt.show()  
                plt.close() 
            is_weight = ( is_weight == False )
            paraNo+=1
 
    def save_out_history(self):
        is_weight = True
        layerNo = 1
        para_w =[]
        para_b =[]
        for p in self.params_:
            if is_weight:
                w = p.flatten()
                para_w.append(w)
            else:
                b = p.flatten()
                para_b.append(b)
                layerNo +=1
            is_weight = ( is_weight == False )
        
        
        f = open('Data_param/Model'+self.modelNo+"_weight.csv","w")
        w = csv.writer(f)
        w.writerows(para_w)
        f.close()
        
        f = open('Data_param/Model'+self.modelNo+"_bias.csv","w")
        w = csv.writer(f)
        w.writerows(para_b)
        f.close()  

        f = open('Data_history/Model'+self.modelNo+"_history.csv","w")
        w = csv.writer(f)
        w.writerows([self.training_history])
        w.writerows([self.test_history])
        f.close()
      
        
            