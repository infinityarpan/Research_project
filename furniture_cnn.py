#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SUITABLE DATASET CREATION FROM INITIAL DATASET

import os
import cv2

path = "/home/arpan/Downloads/Bonn_Furniture_Styles_Dataset/houzz"
os.mkdir("/home/arpan/Downloads/output")
out_path = "/home/arpan/Downloads/output"
# itterate through the folders like bed, lamp, etc.
for f in os.listdir(path):
    os.mkdir(os.path.join(out_path, f))
    d = os.path.join(out_path, f)
    j = os.path.join(path, f)
# itterate through the subfolders of the folders like comtemporary, rustic, etc.
    for ff in os.listdir(j):
        jj = os.path.join(j, ff)
# itterate through the images in the subfolders
        for fff in os.listdir(jj):
# immages with .jpg extension is used for reading with opencv
            if fff.split(".")[1] == 'jpg' :
                print ('jpg')
                jjj = os.path.join(jj, fff)
                img_bgr = cv2.imread(jjj)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                dd = os.path.join(d, fff)
                cv2.imwrite(dd, img_rgb)
print ("done")


# In[ ]:


# SPLITTING THE DATASET INTO TRAIN, TEST AND VALIDATION SET

import splitfolders
inp_folder = "/home/arpan/Downloads/output"
os.mkdir("/home/arpan/Downloads/cnn_imagedata")
out_folder = "/home/arpan/Downloads/cnn_imagedata" 
splitfolders.ratio(inp_folder, out_folder, seed = 42, ratio = (.6, .2, .2))


# In[4]:


# import the required libraries

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# path for the test, train and validation data

train_dir = "/home/arpan/Downloads/cnn_imagedata/train"
test_dir = "/home/arpan/Downloads/cnn_imagedata/test"

# using transfer learning with the pretrained inception v3 model with imagenet weights.
# fixing the weight of the lower layers and we dont want to change this

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')
for layer in pre_trained_model.layers:
    layer.trainable = False
    
# setting parameters to stop the execution once desired accuracy is reached
    
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.959):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

# modifying the top layers for six output classes i.e. bed, chair, table, lamp, sofa and dresser
            
x = layers.Flatten()(pre_trained_model.output)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(6, activation='sigmoid')(x)

# compiling the model, while we could also select other optimizers and choose the best suited, while loss is categorical_crossentropy, as we are dealing with multi class data

model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# augumenting the image data for better learning

train_datagen = ImageDataGenerator(rescale = 1/255, rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1/255)
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'categorical', target_size = (150, 150))
validation_generator = test_datagen.flow_from_directory(test_dir, batch_size = 20, class_mode = 'categorical', target_size = (150, 150))
callbacks = myCallback()
history = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 100, validation_steps = 50, verbose = 2, callbacks=[callbacks])


# In[14]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[19]:


val_dir = "/home/arpan/Downloads/cnn_imagedata/val"

datagen_validation = ImageDataGenerator(rescale = 1/255)
validation_generator = datagen_validation.flow_from_directory(val_dir,
                                                    target_size=(150, 150),
                                                    class_mode='categorical')


# In[12]:


import matplotlib.pyplot as plt

plt.figure(figsize=(30,10))
plt.subplot(1, 2, 1)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.show()


# In[20]:


import numpy as np 
import sklearn

predictions = model.predict_generator(generator=validation_generator)
y_pred = [np.argmax(pb) for pb in predictions]
y_test = validation_generator.classes
class_names = validation_generator.class_indices.keys()

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Normalized confusion matrix')
plt.show()


# In[ ]:




