
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import csv
import time


# ## Import Data

# In[2]:




# ## Graph Construction

# You should decide the following variable:
# - number of hidden layer, 
# - number of hidden unit, 
# - learning rate, 
# - number of iteration 
# - mini-batch size. 
# 
# You have to show your :
# - learning curve, 
# - train error rate 
# - test error rate

# In[3]:

class ModelBase:
    def __init__(self, args, modelNo=0 ):
        # setting of model information (save in 'args')
        self.args = args
        self.modelNo = 'Model_'+str(modelNo)
        
        # init setting of history saving  
        self.train_loss_history = [] # loss_pre_epoch
        self.train_acc_history = [] # acc_pre_epoch
        self.test_loss_history = [] # loss_pre_epoch
        self.test_acc_history = [] # acc_pre_epoch
        self.file_path = "./"+self.modelNo+"/" #create a new folder for file saving
        directory = os.path.dirname(self.file_path)
        try:
            os.stat(directory)
            print ('folder '+directory+' exists !!')
        except:
            os.makedirs(directory)       
            print ('create a new folder: '+directory+' !!')
        
        #placeholder
        self.data = tf.placeholder(tf.float32,[None , 80 , 128]) #維度: data_size()*seq_length(80)*word_vector(128)
        self.target = tf.placeholder(tf.int32,[None])            #維度: data_size()*1 
        
        #model
        self.outputs = self.model(self.data)
        logits = slim.fully_connected(self.outputs[:,-1,:],2,activation_fn=None) 
            # 維度: data_size()*nb_hidden_unit , 取出最後一個output sequence 
            #最後一層, 維度降到2, 值為weight, 視為與機率值成正比 ,維度: data_size()*2
    
        #算accuracy(預測正確率)
        predict = tf.cast(tf.arg_max(logits,1),tf.int32)#取出機率最大的index,令其dtype=int32
        self.acc = tf.contrib.metrics.accuracy(predict , self.target) 
                #https://www.tensorflow.org/api_docs/python/tf/contrib/metrics/accuracy
            
        #算loss(最小化objective function),使用softmax_cross_entropy
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.target, logits)
        
        #定義Optimizer(回傳gredient的方法)
        optim = tf.train.GradientDescentOptimizer(learning_rate=self.args.learning_rate) 
        self.train_op = slim.learning.create_train_op(self.loss,optim)
        
        #Graph Build up
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print ('create a new model: '+self.modelNo+' !!')
        self.args.print_()
        

    def model(self, inputs):
        pass
    
    def train_n_test(self):
        imdb = np.load('imdb_word_emb.npz')
        X_train = imdb['X_train']  # shape = (25000, 80, 128)
        y_train = imdb['y_train']  # shape = (25000,)
        X_test  = imdb['X_test']  # shape = (25000, 80, 128)
        y_test  = imdb['y_test']  # shape = (25000,)
        start_time = time.time()
        batch_size = self.args.batch_size
        for epoch in range(self.args.epochs):
            loss_pre_batch = []
            acc_pre_batch = []
            # Train
            for batch_No in range(int(25000 / batch_size)):
                batch_data = X_train[batch_No*batch_size:(batch_No+1)*batch_size]
                batch_target = y_train[batch_No*batch_size:(batch_No+1)*batch_size]

                _ , loss_, acc_ = self.sess.run([self.train_op , self.loss, self.acc] , 
                                           feed_dict={self.data:batch_data,
                                                      self.target:batch_target})

                loss_pre_batch.append(loss_)
                acc_pre_batch.append(acc_)
            self.train_loss_history.append(np.mean(loss_pre_batch))
            self.train_acc_history.append(np.mean(acc_pre_batch))

            print('epo: '+str(epoch)+', train_loss: '+str(self.train_loss_history[-1])+
                  ', train_acc: '+str(self.train_acc_history[-1]))

            # Test      
            test_loss, test_acc = self.sess.run([self.loss, self.acc] , feed_dict={self.data:X_test, self.target:y_test})  
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_acc)

            print('epo: '+str(epoch)+', test_loss: '+str(self.test_loss_history[-1])+
                  ', test_acc: '+str(self.test_acc_history[-1]))  
            
        # save model   
        #saver = tf.train.Saver()
        #save_path = saver.save(self.sess, self.file_path+self.modelNo+".ckpt") 
        #self.sess.close()
	tf.reset_default_graph()
        
        print ('Spend '+str( time.time()-start_time )+' sec for train and test' )
        
        # plot and save
        self.plot_learning_curve()
        self.plot_accuracy_curve()
        self.save_out_history()
           
            
    def plot_learning_curve(self):

        f = plt.figure(1, figsize=(8,3))
        ax = f.add_subplot(111)
        ax.set_title("Learning Curve_"+self.modelNo,fontsize=20)
        #y_max = np.max(self.train_loss_history+self.test_loss_history)+0.1
        y_max = 0.8
        ax.set(ylim=[0,y_max],ylabel='loss',xlabel='epoch')
        ax.plot(range(self.args.epochs) , self.train_loss_history, linewidth=2.0, label='Train')
        ax.plot(range(self.args.epochs) , self.test_loss_history, linewidth=2.0, label='Test')
        ax.legend(loc='best', frameon=True,fontsize=8) #show label
        plt.savefig(self.file_path+"Learning Curve_"+self.modelNo+'.png')
        #plt.show() 
        plt.close()
        
    def plot_accuracy_curve(self):
        f = plt.figure(1, figsize=(8,3))
        ax = f.add_subplot(111)
        ax.set_title("Accuracy Curve_"+self.modelNo,fontsize=20)
        #y_min = np.min(self.train_acc_history+self.test_acc_history)+0.1
        y_min = 0.6
        ax.set(ylim=[y_min,1],ylabel='accuracy',xlabel='epoch')
        ax.plot(range(self.args.epochs) , self.train_acc_history, linewidth=2.0, label='Train')
        ax.plot(range(self.args.epochs) , self.test_acc_history, linewidth=2.0, label='Test')
        ax.legend(loc='best', frameon=True,fontsize=8) #show label
        plt.savefig(self.file_path+"Accuracy Curve_"+self.modelNo+'.png')
        #plt.show() 
        plt.close()

    def save_out_history(self):
        f = open(self.file_path+self.modelNo+"_history.csv","a")
        w = csv.writer(f)
        w.writerows([ ['','','epoch'] + range(1,self.args.epochs+1) ])
        w.writerows([ [self.modelNo,'train','loss'] + self.train_loss_history ])
        w.writerows([ [self.modelNo,'test','loss'] + self.test_loss_history ])
        w.writerows([ [self.modelNo,'train','acc'] + self.train_acc_history ])
        w.writerows([ [self.modelNo,'test','acc'] + self.test_acc_history ])
        f.close()
        


# In[4]:

class MultiRNN(ModelBase):
    def model(self, inputs):       
        # 判斷cell type: 'RNN','LSTM','GRU'
        if self.args.cell_type == 'RNN':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif self.args.cell_type == 'GRU':
            cell_fn = tf.contrib.rnn.GRUCell
        elif self.args.cell_type == 'LSTM':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        #elif self.args.cell_type == 'NAS':
        #    cell_fn = tf.contrib.rnn.NASCell
        else:
            raise Exception("model type not support "+self.args.cell_type)
            
        # Graph construction ( multi-layer RNN )
        cells = []
        for _ in range( self.args.nb_hidden_layers ): 
            cell = cell_fn( self.args.nb_hidden_units )
            cells.append(cell)      
        multi_cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        outputs,states = tf.nn.dynamic_rnn(multi_cells,inputs,dtype=tf.float32)
        return outputs
    


# In[5]:

class args:
    def __init__(self, cell_type='LSTM', epochs=5, nb_hidden_layers=2,
                 nb_hidden_units=200, learning_rate=0.01, batch_size=32): 
        self.cell_type =cell_type #'RNN','LSTM','GRU'
        self.epochs =epochs
        self.nb_hidden_layers =nb_hidden_layers
        self.nb_hidden_units =nb_hidden_units
        self.learning_rate =learning_rate 
        self.batch_size =batch_size  
    def print_(self):
        print ('cell_type',self.cell_type)
        print ('epochs',self.epochs)
        print ('nb_hidden_layers',self.nb_hidden_layers)
        print ('nb_hidden_units',self.nb_hidden_units)
        print ('learning_rate',self.learning_rate)
        print ('batch_size',self.batch_size )






