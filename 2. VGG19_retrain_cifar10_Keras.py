
# coding: utf-8

# to do:
#     - Initial learning rate: 0.01, divide by 10 at 81, 122 epoch
#     - subtract RGB mean value
#     - conv layer init = random_normal(stddev = 0.03), fc layer init = random_normal(stddev = 0.01)

# In[1]:


import numpy as np
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Conv2D
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.initializers import RandomNormal
from keras import backend as K


# ### Data

# In[2]:

batch_size = 128
num_classes = 10
epochs = 164
data_augmentation = True


# In[3]:

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
# data util
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 # norm to [0,1]
x_test /= 255
datagen = ImageDataGenerator(
                            featurewise_center=False,  # set input mean to 0 over the dataset
                            samplewise_center=False,  # set each sample mean to 0
                            featurewise_std_normalization=False,  # divide inputs by std of the dataset
                            samplewise_std_normalization=False,  # divide each input by its std
                            zca_whitening=False,  # apply ZCA whitening
                            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                            horizontal_flip=True,  # randomly flip images
                            vertical_flip=False # randomly flip images
                            )
datagen.fit(x_train)


# ### Model

# In[4]:

base_model = VGG19(weights='imagenet')


# In[5]:

conv_init = RandomNormal(stddev=0.03)
fc_init = RandomNormal(stddev=0.01)


# In[6]:

img_input = Input(shape=(32,32,3))
# get layers from VGG19
x = base_model.layers[1](img_input)
for layer in base_model.layers[2:21]: 
    x = layer(x)
# add layers by ourselves    
x = Flatten()(x)
x = Dense(4096, activation='relu', name='fc6',kernel_initializer=fc_init)(x)
x = Dropout(0.5)(x)
x = base_model.layers[24](x)
x = Dropout(0.5)(x)
predictions =  Dense(10, activation='softmax', name='fc8',kernel_initializer=fc_init)(x)
model = Model(inputs=img_input, outputs=predictions)


# In[7]:

opt = SGD(lr=0.01,momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



# ### Train

# In[9]:

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch= (x_train.shape[0] // batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test) )


# In[ ]:



