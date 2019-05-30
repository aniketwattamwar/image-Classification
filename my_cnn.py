# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:08:30 2018

@author: hp
"""


#importing the keras models and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#iniliasing the CNN
classifier = Sequential()

#step-1 convolution
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation = 'relu'))
#format is diff for theano backend. this is tf backend


#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#flattening
classifier.add(Flatten())

#full connection
classifier.add(Dense(output_dim = 128, activation='relu'))

classifier.add(Dense(output_dim = 1, activation='sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam',loss= 'binary_crossentropy' , metrics = ['accuracy'])

#fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'data',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='sparse')

#test_set = test_datagen.flow_from_directory(
#                                                'dataset/test_set',
#                                                target_size=(64, 64),
#                                                batch_size=32,
#                                                class_mode='sparse')
 
classifier.fit_generator(
                                training_set,
                                steps_per_epoch=50,
                                epochs=3)
#                                validation_data=test_set,
#                                validation_steps=1)
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
#classifier.fit(test_img)
#from IPython import display
#from PIL import Image



import numpy as np
from keras.preprocessing import image
cat_img = image.load_img('cat_image.jpg',target_size = (64,64))

cat_array = np.asarray(cat_img, dtype='int64')
cat_array = np.expand_dims(cat_img, axis = 0)
cat_array = cat_array / 255     

cat_result = classifier.predict(cat_array)
training_set.class_indices
#classes = classifier.predict_classes(cat_array)  
#if result[0][0] >= 0.: 
#    prediction = 'dog'
#else:
#    prediction = 'cat'
#print(prediction)
#
#print(result)
dog_img = image.load_img('dog_image.jpg',target_size = (64,64))

dog_array = np.asarray(dog_img, dtype='int64')
dog_array = np.expand_dims(dog_img, axis = 0)
dog_array = dog_array / 255     

dog_result = classifier.predict(dog_array)
dog_classes = classifier.predict_classes(dog_array)
