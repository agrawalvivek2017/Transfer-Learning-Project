# -*- coding: utf-8 -*-
"""Project2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/174l-dIcNDZ7IJwmuhjR1PAI0mIDpo4PA
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import PIL
import PIL.Image
import tensorflow_datasets as tfds
import pathlib
import cv2
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

from zipfile import ZipFile
file_name = "flowers.zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('Done')

data_dir='./flowers'
data_dir=pathlib.Path(data_dir)
list_flower=list(data_dir.glob('*/*.jpg'))
image_count = len(list_flower)
print(image_count)

batch_size = 32
img_height = 480
img_width = 480
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

validation_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

val_batches = tf.data.experimental.cardinality(validation_ds)
test_dataset = validation_ds.take(val_batches)
validation_dataset = validation_ds.skip(val_batches//15)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
IMG_SHAPE = (img_height, img_width) + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(13)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(480, 480, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
len(model.trainable_variables)

initial_epochs = 5
loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 50

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])
model.summary()
len(model.trainable_variables)

fine_tune_epochs = 5
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)

loss, accuracy = model.evaluate(test_dataset)
print(len(test_dataset))
print('Test accuracy :', accuracy)

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 50

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/100),
              metrics=['accuracy'])
model.summary()
len(model.trainable_variables)

fine_tune_epochs = 3
total_epochs =   fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                  
                         validation_data=validation_dataset)

model.save('VXA200020_model.h5')

loss, accuracy = model.evaluate(test_dataset)
print(len(test_dataset))
print('Test accuracy :', accuracy)

image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
sol = np.zeros(len(predictions))
c=-1
for p in predictions:
  c=c+1
  m = max(p)
  for i in range(0,len(p)):
    if p[i] != m:
      p[i] =0
    else:
      p[i]=1 
      sol[c]=int(i) 

plt.figure(figsize=(10, 10))      
for i in range(len(sol)):
  ax = plt.subplot(len(sol)//5, len(sol)//5, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[int(sol[i])])
  plt.axis("off")

model.test = tf.keras.models.load_model('VXA200020_model.h5')

loss, accuracy = model.test.evaluate(train_dataset)
print(len(train_dataset))
print('Train accuracy :', accuracy)

loss, accuracy = model.test.evaluate(test_dataset)
print(len(test_dataset))
print('Test accuracy :', accuracy)