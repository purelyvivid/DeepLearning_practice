
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


# ### Import Data

# In[2]:

data = loadmat("SVHN.mat")
train_x = data['train_x']
train_label = data['train_label']

test_x = data['test_x']
test_label = data['test_label']


# ### Model Construction

# In[3]:

class Linear:
    def __init__(self,
                 input_size,
                 output_size,
                 active_fn='Relu',
                 L2_regularization_lamda=0):
        
        self.W = np.random.normal(0,0.01 ,size=(input_size,output_size))
        self.b = np.random.randn(output_size)
        self.L2_regularization_lamda = L2_regularization_lamda
        self.cache = None 
        if active_fn is None:
            self.active_fn = lambda x : x
            self.dactive_fn = lambda x : 1
        else:
            self.active_fn = lambda x : np.maximum(0, x)
            self.dactive_fn = lambda x : x > 0
    
    def forward(self,x):
        Z = x.dot(self.W) + self.b
        self.cache = (Z,x)
        return self.active_fn(Z)
    
    def backward(self,dout):  
        Z,x = self.cache
        G = dout * self.dactive_fn(Z)
        din = G.dot(self.W.T)
        dW = x.T.dot(G) + self.L2_regularization_lamda*self.W
        db = G.sum(axis=0)
        return din,dW,db


# In[4]:

def softmax(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


# In[5]:

class CrossEntropyLoss:    
    def forward(self,probs,labels):
        tmp = np.log(probs) * labels
        return -tmp.sum(axis=1).mean()
    
    def backward(self,probs,labels):
        dloss = probs - labels
        return dloss


# In[6]:

class SGD:
    def __init__(self,net,learning_rate):
        self.net = net
        self.learning_rate = learning_rate
    
    def step(self,dloss):
        dparams = self.net.backward(dloss)
        
        for L , dL in zip(self.net.layers,dparams):
            L.W = L.W - self.learning_rate*dL[0]
            L.b = L.b - self.learning_rate*dL[1]


# In[7]:

class ADAM:
    def __init__(self,
                 net,
                 learning_rate=1e-7,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        self.t = 0
        self.net = net
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.exp_avg = []
        self.exp_avg_sq = []
    
    def step(self,dloss):
        dparams = self.net.backward(dloss)
        
        if self.t == 0:
            for dL in dparams:
                self.exp_avg.append((np.zeros_like(dL[0]) , np.zeros_like(dL[1])))                
                self.exp_avg_sq.append((np.zeros_like(dL[0]) , np.zeros_like(dL[1])))
        self.t += 1
        
        #lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
        lr_t = self.learning_rate * math.sqrt(1 - self.betas[1]**self.t) / (1 - self.betas[0]**self.t)
        
        #m_t <- beta1 * m_{t-1} + (1 - beta1) * g
        new_exp_avg = []
        for mu_t,dL in zip(self.exp_avg , dparams):
            mu_t_W = self.betas[0]*mu_t[0] + (1 - self.betas[0])*dL[0]
            mu_t_b = self.betas[0]*mu_t[1] + (1 - self.betas[0])*dL[1]
            new_exp_avg.append((mu_t_W , mu_t_b))
        self.exp_avg = new_exp_avg
        
        #v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
        new_exp_avg_sq = []        
        for v_t,dL in zip(self.exp_avg_sq , dparams):
            v_t_W = self.betas[1]*v_t[0] + (1 - self.betas[1])*dL[0]*dL[0]
            v_t_b = self.betas[1]*v_t[1] + (1 - self.betas[1])*dL[1]*dL[1]
            new_exp_avg_sq.append((v_t_W , v_t_b))
        self.exp_avg_sq = new_exp_avg_sq
        
        #variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
        for L,mt,vt in zip(self.net.layers,self.exp_avg,self.exp_avg_sq):
            mt_W = mt[0] / (1 - self.betas[0]**self.t)
            mt_b = mt[1] / (1 - self.betas[0]**self.t)
            
            vt_W = vt[0] / (1 - self.betas[1]**self.t)
            vt_b = vt[1] / (1 - self.betas[1]**self.t)
            
            L.W = L.W - lr_t*mt_W / (np.sqrt(vt_W)+self.eps)
            L.b = L.b - lr_t*mt_b / (np.sqrt(vt_b)+self.eps)


# In[8]:

def accuracy(probs , labels):
    preds = probs.argmax(axis=1)
    labels = labels.argmax(axis=1)
    
    return np.mean(preds == labels)


# In[9]:

class Net:
    def __init__(self,layers):
        self.layers= layers
    
    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x,softmax(x)
    def backward(self,dloss):
        dNet = []
        for layer in self.layers[::-1]:
            dloss,dW,db = layer.backward(dloss)
            dNet.append((dW,db))
        
        return dNet[::-1]


# In[10]:

class Trainer:
    def __init__(self,net,criterion,optimizer,logging_freq=10):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.logging_freq = logging_freq
        self.train_loss_history = []
        self.train_error_rate_history = []
        self.test_error_rate_history = []
        
        self.epochs = 0
        
    def fit(self,train_data,
                 train_labels,
                 test_data,
                 test_labels,
                 epochs=200,
                 batch_size=100):
        self.epochs = epochs
        train_data_size = train_data.shape[0]
        for epoch in range(epochs):
            for batch in range(train_data_size / batch_size):
                batch_data = train_data[batch*batch_size : (batch+1)*batch_size]
                batch_labels = train_labels[batch*batch_size : (batch+1)*batch_size]
                
                logits,probs = self.net.forward(batch_data)
                self.optimizer.step(self.criterion.backward(probs,batch_labels))
            
            logits,probs = self.net.forward(train_data)
            train_loss = self.criterion.forward(probs,train_labels)
            train_error_rate = 1-accuracy(probs , train_labels)
                
            logits,probs = self.net.forward(test_data)
            test_error_rate = 1-accuracy(probs , test_labels)
            self.train_loss_history.append(train_loss)
            
            self.train_error_rate_history.append(train_error_rate)
            self.test_error_rate_history.append(test_error_rate)
            
            if epoch % self.logging_freq == 0:
                print "Epoch:%s, Train_loss:%s, Train_error_rate:%s, Test_error_rate:%s" % (epoch,
                                                                                         train_loss,
                                                                                         train_error_rate,
                                                                                         test_error_rate)
    def predict(self,data):
        return self.net.forward(data)[1].argmax(axis=1)
    
    def plot(self): #simple plot
        plt.figure(1)
        x = range(0,self.epochs)
        
        plt.subplot(211)
        plt.plot(x,self.train_loss_history)
        plt.title("Learning curve")

        plt.subplot(212)
        plt.plot(x,self.train_error_rate_history,'b')
        plt.plot(x,self.test_error_rate_history,'r')
        plt.title("Training and testing error rate")
        plt.show()


# In[55]:

import csv


# In[56]:

class Trainer_abstract: # for a single trainer, retrieve it's history for plotting
    def __init__(self,tr,RegType,OptType): # tr: a single trainer
        self.cache = [ range(0,tr.epochs), tr.train_loss_history ,
                       tr.train_error_rate_history, tr.test_error_rate_history]
        self.x = self.cache[0]
        self.train_loss_history = self.cache[1]
        self.train_error_rate_history  = self.cache[2]
        self.test_error_rate_history = self.cache[3]
        self.trainer_name = RegType + ' + ' + OptType
        self.epochs = tr.epochs
        self.logging_freq = tr.logging_freq
        
    def save_out_history(self): 
        f = open(self.trainer_name+"_history.csv","w")
        w = csv.writer(f)
        w.writerows(self.cache)
        f.close()
        
    def plot(self): #refined plot       
        # draw Learning Curve
        f = plt.figure(1, figsize=(5,3))
        ax = f.add_subplot(111)
        ax.set_title(self.trainer_name+" - Learning Curve",fontsize=15)
        ax.plot(self.x,self.train_loss_history,linewidth=2.0)
        ax.set(ylim =[0,3],xlabel='epoch',ylabel='loss')
        plt.savefig('Learning Curve_'+self.trainer_name+'.png')
        plt.show()  
        
        # draw Error Rate
        f = plt.figure(1, figsize=(5,3))
        ax = f.add_subplot(111)
        ax.set_title(self.trainer_name+" - Error Rate",fontsize=15)
        ax.plot(self.x,self.train_error_rate_history, label='train',linewidth=2.0)
        ax.plot(self.x,self.test_error_rate_history, label='test',linewidth=2.0)
        ax.set(ylim =[0,1],xlabel='epoch',ylabel='Error Rate')
        ax.legend(loc='best', frameon=True,fontsize=8) #show label
        plt.savefig('Error_Rate_'+self.trainer_name+'.png')
        plt.show() 


# ### Model 1

# In[11]:

N_baseline = Net([Linear(784,400),
                  Linear(400,200),
                  Linear(200,100),
                  Linear(100,10,active_fn=None)])
loss = CrossEntropyLoss()
sgd = SGD(N_baseline,1e-3)

trainer_baseline = Trainer(net=N_baseline,
                           criterion=loss,
                           optimizer=sgd,
                           logging_freq=10)

trainer_baseline.fit(train_x,train_label,test_x,test_label,epochs=100)
trainer_baseline.plot()


# In[58]:

trabs1 = Trainer_abstract(trainer_baseline,'without_Regulization','sgd')
trabs1.plot()
trabs1.save_out_history()


# ### Model 2

# In[50]:

N_use_adam = Net([Linear(784,400),
                  Linear(400,200),
                  Linear(200,100),
                  Linear(100,10,active_fn=None)])
loss = CrossEntropyLoss()
adam = ADAM(N_use_adam,1e-3)

trainer_use_adam = Trainer(net=N_use_adam,
                           criterion=loss,
                           optimizer=adam,
                           logging_freq=10)

trainer_use_adam.fit(train_x,train_label,test_x,test_label,epochs=100)
trainer_use_adam.plot()


# In[59]:

trabs2 = Trainer_abstract(trainer_use_adam,'without_Regulization','adam')
trabs2.plot()
trabs2.save_out_history()


# ### Model 3

# In[38]:

N_use_L2 = Net([Linear(784,400),
                Linear(400,200),
                Linear(200,100,L2_regularization_lamda=10),
                Linear(100,10,active_fn=None,L2_regularization_lamda=10)])
loss = CrossEntropyLoss()
sgd = SGD(N_use_L2,2e-6)

trainer_use_L2 = Trainer(net=N_use_L2,
                         criterion=loss,
                         optimizer=sgd,
                         logging_freq=10)

trainer_use_L2.fit(train_x,train_label,test_x,test_label,epochs=100)
trainer_use_L2.plot()


# In[60]:

trabs3 = Trainer_abstract(trainer_use_L2,'L2_Regulization','sgd')
trabs3.plot()
trabs3.save_out_history()


# ### Model 4

# In[35]:

N_use_L2_adam = Net([Linear(784,400),
                     Linear(400,200),
                     Linear(200,100,L2_regularization_lamda=10),
                     Linear(100,10,active_fn=None,L2_regularization_lamda=10)])
loss = CrossEntropyLoss()
adam = ADAM(N_use_L2_adam,1e-3)

trainer_use_L2_adam = Trainer(net=N_use_L2_adam,
                              criterion=loss,
                              optimizer=adam,
                              logging_freq=10)

trainer_use_L2_adam.fit(train_x,train_label,test_x,test_label,epochs=100)
trainer_use_L2_adam.plot()


# In[61]:

trabs4 = Trainer_abstract(trainer_use_L2_adam,'L2_Regulization','adam')
trabs4.plot()
trabs4.save_out_history()


# ### Total Plot

# In[62]:

class Total_Trainer_Plot: # for 4 trainers
    def __init__(self,trainerName,trainerList): #trainerList: trainer1,2,..
        self.trainerName = trainerName
        self.x = []
        self.learning_curve_y = []
        self.train_error_rate_y  = []
        self.test_error_rate_y = []
        for tr in trainerList:
            self.x.append(  range(0,tr.epochs) )
            self.learning_curve_y.append( tr.train_loss_history )
            self.train_error_rate_y.append( tr.train_error_rate_history )
            self.test_error_rate_y.append( tr.test_error_rate_history )
            
        
    def plot(self):
        # draw Learning Curve
        f = plt.figure(1, figsize=(8,3))
        ax = f.add_subplot(111)
        for i in xrange(4):
            ax.set_title("Learning Curve",fontsize=20)
            ax.plot(self.x[i],self.learning_curve_y[i], label=self.trainerName[i],linewidth=2.0)
            ax.set_ylim([0,3])
            ax.set_ylabel='loss'
            ax.legend(loc='best', frameon=True,fontsize=8) #show label
        plt.savefig('Learning_Curve_4in1_trans.png', transparent=True)
        plt.savefig('Learning_Curve_4in1.png')
        plt.show()        
         
        plt.close('all')
        # draw Error Rate
        f, axarr = plt.subplots(2, 2,sharex='col', sharey='row', figsize=(8,5))
        splt = [ axarr[0, 0], axarr[0, 1],axarr[1, 0],axarr[1, 1] ]
        for i in xrange(4):
            splt[i].set_title(self.trainerName[i])
            splt[i].plot(self.x[i],self.train_error_rate_y[i], 'b', label="train",
                         ls='-',linewidth=2.0)
            splt[i].plot(self.x[i],self.test_error_rate_y[i], 'g', label="test",
                         ls='-',linewidth=2.0)
            splt[i].legend(loc='best', frameon=True,fontsize=10) #show label
            splt[i].set(ylim=[0,1],ylabel='Error Rate')
            splt[i].margins(x=0,y=0.1)
        plt.savefig('Error_Rate_4in1_trans.png', transparent=True)
        plt.savefig('Error_Rate_4in1.png')
        plt.show()       

        


# In[63]:

# trainer_abstract plot (4 in 1)
ab_trainerName = [ trabs1.trainer_name, trabs2.trainer_name, 
                trabs3.trainer_name, trabs4.trainer_name ]
ab_trainer_List = [trabs1,trabs2,trabs3,trabs4]
totr = Total_Trainer_Plot (ab_trainerName,ab_trainer_List)
totr.plot()


# In[ ]:



